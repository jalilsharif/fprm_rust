//! CSV / XLSX loaders → `Array2` with rows = time, columns = series.

use std::path::Path;

use calamine::{open_workbook_auto, Data, Reader};
use csv::ReaderBuilder;
use ndarray::Array2;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    #[error("Excel error: {0}")]
    Excel(String),
    #[error("empty table")]
    Empty,
    #[error("ragged row lengths in table")]
    Ragged,
    #[error("non-numeric cell: {0}")]
    NonNumeric(String),
}

fn parse_f64(s: &str) -> Result<f64, IoError> {
    let t = s.trim();
    if t.is_empty() {
        return Ok(f64::NAN);
    }
    t.parse::<f64>()
        .map_err(|_| IoError::NonNumeric(t.to_string()))
}

/// Load CSV. If the first row looks like integer indices (`1,2,3,...`), it is treated as a header and skipped.
pub fn load_csv(path: impl AsRef<Path>) -> Result<Array2<f64>, IoError> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_path(path)?;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let vals: Result<Vec<f64>, _> = rec.iter().map(parse_f64).collect();
        rows.push(vals?);
    }
    if rows.is_empty() {
        return Err(IoError::Empty);
    }
    // Drop header of consecutive integers 1..ncols if present
    if looks_like_index_header(&rows[0]) {
        rows.remove(0);
    }
    table_to_array(rows)
}

fn looks_like_index_header(row: &[f64]) -> bool {
    if row.is_empty() {
        return false;
    }
    row.iter()
        .enumerate()
        .all(|(i, v)| (*v - (i as f64 + 1.0)).abs() < 1e-9)
}

fn table_to_array(rows: Vec<Vec<f64>>) -> Result<Array2<f64>, IoError> {
    if rows.is_empty() {
        return Err(IoError::Empty);
    }
    let ncols = rows[0].len();
    if ncols == 0 {
        return Err(IoError::Empty);
    }
    if rows.iter().any(|r| r.len() != ncols) {
        return Err(IoError::Ragged);
    }
    let nrows = rows.len();
    let mut a = Array2::<f64>::zeros((nrows, ncols));
    for i in 0..nrows {
        for j in 0..ncols {
            a[(i, j)] = rows[i][j];
        }
    }
    Ok(a)
}

/// Load first sheet of an Excel workbook (`.xlsx` / `.xls`).
pub fn load_xlsx(path: impl AsRef<Path>) -> Result<Array2<f64>, IoError> {
    let mut workbook = open_workbook_auto(path).map_err(|e| IoError::Excel(e.to_string()))?;
    let sheet_names = workbook.sheet_names().to_vec();
    let name = sheet_names
        .first()
        .ok_or_else(|| IoError::Excel("no sheets".into()))?
        .clone();
    let range = workbook
        .worksheet_range(&name)
        .map_err(|e| IoError::Excel(e.to_string()))?;

    let mut rows: Vec<Vec<f64>> = Vec::new();
    for row in range.rows() {
        let mut vals = Vec::with_capacity(row.len());
        for cell in row {
            let v = match cell {
                Data::Float(f) => *f,
                Data::Int(i) => *i as f64,
                Data::Empty => f64::NAN,
                Data::String(s) => parse_f64(s)?,
                other => {
                    // Bool / Error / Date — try string form
                    let s = format!("{other}");
                    parse_f64(&s).unwrap_or(f64::NAN)
                }
            };
            vals.push(v);
        }
        if vals.iter().all(|v| v.is_nan()) {
            continue;
        }
        rows.push(vals);
    }
    if rows.is_empty() {
        return Err(IoError::Empty);
    }
    if looks_like_index_header(&rows[0]) {
        rows.remove(0);
    }
    // Pad ragged Excel rows with NaN to max width
    let ncols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    for r in &mut rows {
        r.resize(ncols, f64::NAN);
    }
    table_to_array(rows)
}

/// Dispatch on extension.
pub fn load_table(path: impl AsRef<Path>) -> Result<Array2<f64>, IoError> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "csv" | "txt" => load_csv(path),
        "xlsx" | "xls" | "xlsm" | "xlsb" | "ods" => load_xlsx(path),
        _ => {
            // try CSV first, then Excel
            load_csv(path).or_else(|_| load_xlsx(path))
        }
    }
}

pub fn write_csv(path: impl AsRef<Path>, data: &Array2<f64>) -> Result<(), IoError> {
    let mut wtr = csv::Writer::from_path(path)?;
    for row in data.rows() {
        let rec: Vec<String> = row.iter().map(|v| format!("{v}")).collect();
        wtr.write_record(&rec)?;
    }
    wtr.flush()?;
    Ok(())
}
