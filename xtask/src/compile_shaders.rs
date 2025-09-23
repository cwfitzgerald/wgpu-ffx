use anyhow::Result;
use camino::{Utf8Path, Utf8PathBuf};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::process::Command;

pub const HELP: &str = r#"xtask-compile-shaders
Compile shaders for the project

USAGE:
    xtask compile-shaders [OPTIONS]

OPTIONS:
    -h, --help    Print help information
"#;

const SHADER_DIRECTORY: &str = "../FidelityFX-SDK-v1.1.4/sdk/src/backends/vk/shaders/fsr3upscaler";
const INCLUDE_DIR: &str = "../FidelityFX-SDK-v1.1.4/sdk/include/FidelityFX/gpu";

fn get_output_directory() -> Result<Utf8PathBuf> {
    Ok(Utf8PathBuf::from("shaders/compiled"))
}

#[derive(Debug, Deserialize)]
struct PermutationConfig {
    base: HashMap<String, toml::Value>,
    permutations: HashMap<String, Vec<toml::Value>>,
}

#[derive(Debug, Clone)]
struct ShaderPermutation {
    shader_file: Utf8PathBuf,
    permutation_id: String, // For tracking which permutation this represents
    defines: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
struct CompilationResult {
    permutation_id: String,
    content_hash: String,
    output_path: Utf8PathBuf,
    size_bytes: usize,
}

#[derive(Debug)]
struct DeduplicationInfo {
    unique_hashes: usize,
    duplicates_eliminated: usize,
    space_saved_mb: f64,
}

pub fn compile_shaders(mut args: pico_args::Arguments) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        print!("{}", HELP);
        return Ok(());
    }

    // Check for unexpected arguments
    let remaining = args.finish();
    if !remaining.is_empty() {
        return Err(anyhow::anyhow!("Unexpected arguments: {:?}", remaining));
    }

    println!("Compiling shaders...");

    // Load permutation configuration
    let perm_config = load_permutation_config()?;

    // Find all GLSL files in the shader directory
    let glsl_files = find_glsl_files()?;

    if glsl_files.is_empty() {
        println!("No GLSL files found in {}", SHADER_DIRECTORY);
        return Ok(());
    }

    // Generate all permutations for all shader files
    let all_permutations = generate_all_permutations(&glsl_files, &perm_config)?;

    println!("Found {} GLSL files", glsl_files.len());
    println!("Generating {} total permutations", all_permutations.len());

    // Set up progress bar
    let progress = ProgressBar::new(all_permutations.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")?
            .progress_chars("##-"),
    );

    // Compile all permutations in parallel and collect results
    let compilation_results: Vec<Result<CompilationResult>> = all_permutations
        .par_iter()
        .map(|permutation| {
            let result = compile_single_permutation(permutation);
            progress.inc(1);
            result
        })
        .collect();

    progress.finish_with_message("Compilation complete!");

    // Process results and perform deduplication
    let mut successful_results = Vec::new();
    let mut error_count = 0;

    for (i, result) in compilation_results.iter().enumerate() {
        match result {
            Ok(compilation_result) => {
                successful_results.push(compilation_result.clone());
            }
            Err(e) => {
                eprintln!(
                    "Failed to compile permutation {}: {}",
                    all_permutations[i].permutation_id, e
                );
                error_count += 1;
            }
        }
    }

    if error_count > 0 {
        eprintln!("{} shader compilations failed", error_count);
    }

    // Perform deduplication analysis
    let dedup_info = analyze_deduplication(&successful_results);

    println!("Compilation and deduplication complete!");
    println!("Original permutations: {}", all_permutations.len());
    println!("Successful compilations: {}", successful_results.len());
    println!("Unique shader variants: {}", dedup_info.unique_hashes);
    println!(
        "Duplicates eliminated: {}",
        dedup_info.duplicates_eliminated
    );
    println!("Space saved: {:.2} MB", dedup_info.space_saved_mb);

    if error_count > 0 {
        return Err(anyhow::anyhow!(
            "{} shader compilations failed",
            error_count
        ));
    }

    Ok(())
}

fn load_permutation_config() -> Result<PermutationConfig> {
    let config_path = "shaders/fsr3upscaler/perm.toml";
    let config_content = fs::read_to_string(config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", config_path, e))?;

    let config: PermutationConfig = toml::from_str(&config_content)
        .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", config_path, e))?;

    Ok(config)
}

fn find_glsl_files() -> Result<Vec<Utf8PathBuf>> {
    let shader_dir = Utf8Path::new(SHADER_DIRECTORY);

    if !shader_dir.exists() {
        return Err(anyhow::anyhow!(
            "Shader directory does not exist: {}",
            SHADER_DIRECTORY
        ));
    }

    let mut glsl_files = Vec::new();

    for entry in fs::read_dir(shader_dir)? {
        let entry = entry?;
        let path = Utf8PathBuf::try_from(entry.path())
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in path: {}", e))?;

        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension == "glsl" {
                    glsl_files.push(path);
                }
            }
        }
    }

    Ok(glsl_files)
}

fn generate_all_permutations(
    glsl_files: &[Utf8PathBuf],
    config: &PermutationConfig,
) -> Result<Vec<ShaderPermutation>> {
    let mut all_permutations = Vec::new();

    // Generate the cartesian product of all permutation values
    let permutation_keys: Vec<&String> = config.permutations.keys().collect();
    let permutation_values: Vec<&Vec<toml::Value>> = permutation_keys
        .iter()
        .map(|key| config.permutations.get(*key).unwrap())
        .collect();

    let combinations = generate_cartesian_product(&permutation_values);

    for shader_file in glsl_files {
        for combination in &combinations {
            let mut defines = Vec::new();

            // Add base defines
            for (key, value) in &config.base {
                let value_str = toml_value_to_string(value);
                defines.push((key.clone(), value_str));
            }

            // Add permutation defines
            for (i, value) in combination.iter().enumerate() {
                let key = permutation_keys[i].clone();
                let value_str = toml_value_to_string(value);
                defines.push((key, value_str));
            }

            // Generate permutation ID for tracking
            let mut permutation_id = shader_file.file_stem().unwrap().to_string();
            for value in combination {
                permutation_id.push('_');
                permutation_id.push_str(&toml_value_to_string(value));
            }

            all_permutations.push(ShaderPermutation {
                shader_file: shader_file.clone(),
                permutation_id,
                defines,
            });
        }
    }

    Ok(all_permutations)
}

fn toml_value_to_string(value: &toml::Value) -> String {
    match value {
        toml::Value::Integer(i) => i.to_string(),
        toml::Value::String(s) => s.clone(),
        toml::Value::Boolean(b) => {
            if *b {
                "1".to_string()
            } else {
                "0".to_string()
            }
        }
        toml::Value::Float(f) => f.to_string(),
        _ => value.to_string(),
    }
}

fn generate_cartesian_product(arrays: &[&Vec<toml::Value>]) -> Vec<Vec<toml::Value>> {
    if arrays.is_empty() {
        return vec![vec![]];
    }

    let mut result = vec![vec![]];

    for array in arrays {
        let mut new_result = Vec::new();
        for combination in &result {
            for value in *array {
                let mut new_combination = combination.clone();
                new_combination.push(value.clone());
                new_result.push(new_combination);
            }
        }
        result = new_result;
    }

    result
}

fn format_command(cmd: &Command) -> String {
    let program = cmd.get_program().to_string_lossy();
    let args: Vec<String> = cmd
        .get_args()
        .map(|arg| arg.to_string_lossy().to_string())
        .collect();
    format!("{} {}", program, args.join(" "))
}

fn compile_single_permutation(permutation: &ShaderPermutation) -> Result<CompilationResult> {
    // Build the glslc command
    let mut cmd = Command::new("glslc");
    cmd.arg("--target-env=vulkan1.2");
    // Add include directory
    cmd.arg(format!("-I{}", INCLUDE_DIR));

    // Add defines for this permutation
    for (key, value) in &permutation.defines {
        cmd.arg(format!("-D{}={}", key, value));
    }

    // Set shader stage
    cmd.arg("-fshader-stage=comp");

    // Input and output files (compile to memory first)
    cmd.arg(&permutation.shader_file);
    cmd.arg("-o");
    cmd.arg("-"); // Output to stdout

    // Set optimization level
    cmd.arg("-O");

    // Execute the command
    let output = cmd.output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let command_str = format_command(&cmd);
        return Err(anyhow::anyhow!(
            "glslc failed for {}:\nCommand: {}\nError: {}",
            permutation.permutation_id,
            command_str,
            stderr
        ));
    }

    // Calculate hash of the compiled output
    let mut hasher = DefaultHasher::new();
    output.stdout.hash(&mut hasher);
    let hash = hasher.finish();
    let hash_hex = format!("{:016x}", hash);

    // Get the base shader name (without extension)
    let shader_name = permutation.shader_file.file_stem().unwrap_or("unknown");

    // Create output filename based on shader name and hash
    let output_path =
        get_output_directory()?.join(format!("{}_{}.spv", shader_name, &hash_hex[..8]));

    // Write the compiled shader to disk
    std::fs::create_dir_all(output_path.parent().unwrap())?;
    std::fs::write(&output_path, &output.stdout)?;

    Ok(CompilationResult {
        permutation_id: permutation.permutation_id.clone(),
        content_hash: hash_hex,
        output_path,
        size_bytes: output.stdout.len(),
    })
}

fn analyze_deduplication(results: &[CompilationResult]) -> DeduplicationInfo {
    use std::collections::HashMap;

    let mut hash_groups: HashMap<String, Vec<&CompilationResult>> = HashMap::new();

    // Group results by content hash
    for result in results {
        hash_groups
            .entry(result.content_hash.clone())
            .or_default()
            .push(result);
    }

    let unique_hashes = hash_groups.len();
    let total_permutations = results.len();
    let duplicates_eliminated = total_permutations - unique_hashes;

    // Calculate space saved
    let mut total_duplicate_bytes = 0;
    for group in hash_groups.values() {
        if group.len() > 1 {
            // Count extra copies beyond the first one
            for i in 1..group.len() {
                total_duplicate_bytes += group[i].size_bytes;
            }
        }
    }

    let space_saved_mb = total_duplicate_bytes as f64 / (1024.0 * 1024.0);

    DeduplicationInfo {
        unique_hashes,
        duplicates_eliminated,
        space_saved_mb,
    }
}
