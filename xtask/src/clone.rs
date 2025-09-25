//! Download and extract FidelityFX SDK
//!
//! This module handles downloading the FidelityFX SDK from GitHub releases
//! and extracting it to the ffx directory in the repository root.

use anyhow::{Context, Result};
use std::{fs, path::Path, process::Command};

pub const HELP: &str = r#"xtask-clone
Download and extract FidelityFX SDK

USAGE:
    xtask clone [OPTIONS]

OPTIONS:
    -h, --help    Print help information
"#;

const SDK_URL: &str = "https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/download/v1.1.4/FidelityFX-SDK-v1.1.4.zip";
const ZIP_FILENAME: &str = "FidelityFX-SDK-v1.1.4.zip";
const EXTRACT_DIR: &str = "ffx";

pub fn clone(mut args: pico_args::Arguments) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        print!("{HELP}");
        return Ok(());
    }

    // Check for remaining arguments
    if let Some(arg) = args.finish().first() {
        anyhow::bail!("Unexpected argument: {arg:?}");
    }

    println!("Downloading FidelityFX SDK...");
    download_sdk()?;

    println!("Extracting FidelityFX SDK...");
    extract_sdk()?;

    println!("Cleaning up...");
    cleanup_zip()?;

    println!(
        "FidelityFX SDK successfully downloaded and extracted to '{EXTRACT_DIR}'"
    );

    Ok(())
}

fn download_sdk() -> Result<()> {
    // Check if curl is available
    let curl_check = Command::new("curl").arg("--version").output();

    if curl_check.is_err() {
        anyhow::bail!("curl is not available. Please install curl and ensure it's in your PATH.");
    }

    let status = Command::new("curl")
        .arg("-L") // Follow redirects
        .arg("-o")
        .arg(ZIP_FILENAME)
        .arg(SDK_URL)
        .status()
        .context("Failed to execute curl command")?;

    if !status.success() {
        anyhow::bail!(
            "Failed to download FidelityFX SDK. Please check your internet connection and try again."
        );
    }

    // Verify the file was downloaded
    if !Path::new(ZIP_FILENAME).exists() {
        anyhow::bail!("Download appeared to succeed but zip file was not found");
    }

    Ok(())
}

fn extract_sdk() -> Result<()> {
    // Remove existing ffx directory if it exists
    if Path::new(EXTRACT_DIR).exists() {
        fs::remove_dir_all(EXTRACT_DIR)
            .with_context(|| format!("Failed to remove existing '{EXTRACT_DIR}' directory"))?;
    }

    // Check if 7z is available
    let seven_zip_check = Command::new("7z").output();

    if seven_zip_check.is_err() {
        anyhow::bail!("7z is not available. Please install 7-Zip and ensure it's in your PATH.");
    }

    let status = Command::new("7z")
        .arg("x") // Extract with full paths
        .arg(ZIP_FILENAME)
        .arg(format!("-o{EXTRACT_DIR}")) // Output directory
        .status()
        .context("Failed to execute 7z extraction command")?;

    if !status.success() {
        anyhow::bail!("Failed to extract FidelityFX SDK zip file");
    }

    // Verify extraction worked
    if !Path::new(EXTRACT_DIR).exists() {
        anyhow::bail!("Extraction appeared to succeed but ffx directory was not created");
    }

    Ok(())
}

fn cleanup_zip() -> Result<()> {
    if Path::new(ZIP_FILENAME).exists() {
        fs::remove_file(ZIP_FILENAME).context("Failed to clean up downloaded zip file")?;
    }
    Ok(())
}
