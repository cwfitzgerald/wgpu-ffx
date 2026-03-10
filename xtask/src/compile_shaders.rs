//! Shader compilation and permutation generation for FidelityFX shaders.
//!
//! This module handles the compilation of GLSL shaders from the FidelityFX SDK into SPIR-V,
//! with support for shader permutations based on preprocessor defines. The system works as follows:
//!
//! 1. **Discovery**: Find all `perm.toml` configuration files in the `shaders/` directory
//! 2. **Permutation Generation**: For each shader and each combination of define values,
//!    generate a unique shader variant
//! 3. **Compilation**: Compile all permutations to SPIR-V using `glslc`
//! 4. **Deduplication**: Hash-based deduplication eliminates identical compiled outputs
//! 5. **Code Generation**: Generate Rust code with embedded shaders and selection functions
//!
//! ## Configuration Format
//!
//! Each `perm.toml` file defines:
//! - `path.subdirectory`: Path within the FidelityFX SDK to find GLSL files
//! - `base`: Base preprocessor defines applied to all permutations
//! - `permutations`: Map of define names to arrays of possible values
//!
//! ## Output
//!
//! For each configuration, generates:
//! - Compiled SPIR-V files with hash-based names for deduplication
//! - `shaders.rs` with embedded shader data and selection functions

use anyhow::Result;
use camino::{Utf8Path, Utf8PathBuf};
use indexmap::{IndexMap, IndexSet};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Deserialize;
use std::{
    collections::hash_map::DefaultHasher,
    fs,
    hash::{Hash, Hasher},
    process::Command,
};

pub const HELP: &str = r#"xtask-compile-shaders
Compile shaders for the project

USAGE:
    xtask compile-shaders [OPTIONS]

OPTIONS:
    -h, --help    Print help information
"#;

// Configuration constants
const SDK_BASE_PATH: &str = "shaders/src";
const INCLUDE_DIR: &str = "shaders/include";
const SHADERS_DIR: &str = "wgpu-ffx-shaders-spv/src";
const PERM_CONFIG_FILE: &str = "perm.toml";
const GENERATED_FILE_NAME: &str = "mod.rs";

// Hash truncation length for shader file names (8 hex chars = 32 bits)
const HASH_TRUNCATE_LEN: usize = 8;

#[derive(Debug, Deserialize)]
struct ShaderPathConfig {
    subdirectory: String,
}

/// Configuration for shader permutation generation from perm.toml files
#[derive(Debug, Deserialize)]
struct ShaderPermutationConfig {
    path: ShaderPathConfig,
    /// Base preprocessor defines applied to all shader variants
    base: IndexMap<String, toml::Value>,
    /// Map of define names to their possible values for permutation generation
    permutations: IndexMap<String, Vec<toml::Value>>,
}

/// A shader configuration with its source and output locations
#[derive(Debug)]
struct ShaderConfig {
    config: ShaderPermutationConfig,
    /// Directory in the FidelityFX SDK containing GLSL source files
    sdk_shader_directory: Utf8PathBuf,
    /// Directory where compiled shaders and generated code will be written
    output_directory: Utf8PathBuf,
}

#[derive(Debug, Clone)]
struct ShaderPermutation {
    shader_file: Utf8PathBuf,
    permutation_id: String, // For tracking which permutation this represents
    defines: Vec<(String, String)>,
    output_directory: Utf8PathBuf, // Where to place compiled shader
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
        print!("{HELP}");
        return Ok(());
    }

    // Check for unexpected arguments
    let remaining = args.finish();
    if !remaining.is_empty() {
        return Err(anyhow::anyhow!("Unexpected arguments: {remaining:?}"));
    }

    println!("Compiling shaders...");

    // Discover all shader configurations
    let shader_configs = discover_shader_configs()?;

    if shader_configs.is_empty() {
        println!("No perm.toml files found in shaders/ directory");
        return Ok(());
    }

    println!("Found {} shader configurations", shader_configs.len());

    // Clean up old .spv files before compilation
    clean_old_spv_files(&shader_configs)?;

    // Generate all shader permutations
    let (all_permutations, total_glsl_files) = generate_all_shader_permutations(&shader_configs)?;

    println!("Found {total_glsl_files} GLSL files across all configurations");

    // Compile all permutations
    let successful_results = compile_all_permutations(&all_permutations)
        .ok_or(anyhow::anyhow!("Failed to compile shaders"))?;

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

    // Generate Rust embedding code for each shader configuration
    for shader_config in &shader_configs {
        generate_rust_embedding(shader_config, &all_permutations, &successful_results)?;
    }

    Ok(())
}

/// Discover and generate all shader permutations from configurations
fn generate_all_shader_permutations(
    shader_configs: &[ShaderConfig],
) -> Result<(Vec<ShaderPermutation>, usize)> {
    let mut all_permutations = Vec::new();
    let mut total_glsl_files = 0;

    for shader_config in shader_configs {
        let glsl_files = find_glsl_files(&shader_config.sdk_shader_directory)?;
        total_glsl_files += glsl_files.len();

        if glsl_files.is_empty() {
            println!(
                "No GLSL files found in {}",
                shader_config.sdk_shader_directory
            );
            continue;
        }

        let permutations = generate_all_permutations(
            &glsl_files,
            &shader_config.config,
            &shader_config.output_directory,
        )?;
        all_permutations.extend(permutations);
    }

    Ok((all_permutations, total_glsl_files))
}

/// Compile all shader permutations and return results with error count
fn compile_all_permutations(
    all_permutations: &[ShaderPermutation],
) -> Option<Vec<CompilationResult>> {
    println!("Generating {} total permutations", all_permutations.len());

    // Set up progress bar
    let progress = ProgressBar::new(all_permutations.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    // Compile all permutations in parallel and collect results
    let compilation_results: Option<Vec<CompilationResult>> = all_permutations
        .par_iter()
        .map(|permutation| {
            let result = compile_single_permutation(permutation);
            progress.inc(1);
            match result {
                Ok(c) => Some(c),
                Err(e) => {
                    eprintln!(
                        "Error compiling permutation {}: {}",
                        permutation.permutation_id, e
                    );
                    None
                }
            }
        })
        .collect();

    progress.finish_with_message("Compilation complete!");

    compilation_results
}

fn load_shader_config(config_path: &Utf8Path) -> Result<ShaderPermutationConfig> {
    let config_content = fs::read_to_string(config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read shader config {config_path}: {e}"))?;

    let config: ShaderPermutationConfig = toml::from_str(&config_content)
        .map_err(|e| anyhow::anyhow!("Failed to parse shader config {config_path}: {e}"))?;

    Ok(config)
}

fn discover_shader_configs() -> Result<Vec<ShaderConfig>> {
    let shaders_dir = Utf8Path::new(SHADERS_DIR);

    if !shaders_dir.exists() {
        return Err(anyhow::anyhow!(
            "Shaders configuration directory not found: {shaders_dir}\n\
            Please create the shaders/ directory and add perm.toml configuration files."
        ));
    }

    let mut shader_configs = Vec::new();

    // Walk through subdirectories in shaders/
    for entry in fs::read_dir(shaders_dir)? {
        let entry = entry?;
        let path = Utf8PathBuf::try_from(entry.path())
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in shader directory path: {e}"))?;

        if path.is_dir() {
            let perm_file = path.join(PERM_CONFIG_FILE);
            if perm_file.exists() {
                let config = load_shader_config(&perm_file)?;

                // Build SDK shader directory path
                let sdk_shader_directory =
                    Utf8PathBuf::from(SDK_BASE_PATH).join(&config.path.subdirectory);

                // Output directory is the same as the perm.toml directory
                let output_directory = path.clone();

                shader_configs.push(ShaderConfig {
                    config,
                    sdk_shader_directory,
                    output_directory,
                });
            }
        }
    }

    Ok(shader_configs)
}

/// Clean up all existing .spv files in output directories before compilation
fn clean_old_spv_files(shader_configs: &[ShaderConfig]) -> Result<()> {
    println!("Cleaning up old .spv files...");
    let mut total_removed = 0;

    for shader_config in shader_configs {
        let output_dir = &shader_config.output_directory;

        if !output_dir.exists() {
            continue;
        }

        // Find all .spv files in the output directory
        for entry in fs::read_dir(output_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file()
                && let Some(extension) = path.extension()
                && extension == "spv"
            {
                fs::remove_file(&path)?;
                total_removed += 1;
            }
        }
    }

    if total_removed > 0 {
        println!("Removed {total_removed} old .spv files");
    }

    Ok(())
}

fn find_glsl_files(shader_dir: &Utf8Path) -> Result<Vec<Utf8PathBuf>> {
    if !shader_dir.exists() {
        return Err(anyhow::anyhow!(
            "FidelityFX shader directory not found: {shader_dir}\n\
            Please ensure the FidelityFX SDK is present at the expected location."
        ));
    }

    let mut glsl_files = Vec::new();

    for entry in fs::read_dir(shader_dir)? {
        let entry = entry?;
        let path = Utf8PathBuf::try_from(entry.path())
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in shader file path: {e}"))?;

        if path.is_file()
            && let Some(extension) = path.extension()
            && extension == "glsl"
        {
            glsl_files.push(path);
        }
    }

    Ok(glsl_files)
}

fn generate_all_permutations(
    glsl_files: &[Utf8PathBuf],
    config: &ShaderPermutationConfig,
    output_directory: &Utf8Path,
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
                output_directory: output_directory.to_path_buf(),
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

/// Convert shader parameter name like "FFX_HALF" to Rust enum name like "Half"
fn shader_param_to_enum_name(param_name: &str) -> String {
    param_name
        .strip_prefix("FFX_")
        .unwrap_or(param_name)
        .to_lowercase()
        .split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect::<String>()
}

/// Convert shader parameter name like "FFX_HALF" to function parameter name like "half"
fn shader_param_to_function_param(param_name: &str) -> String {
    param_name.to_lowercase().replace("ffx_", "")
}

/// Generates the cartesian product of multiple arrays.
///
/// The cartesian product creates all possible combinations by taking one element
/// from each input array. For example:
///
/// Input: [[A, B], [1, 2]]
/// Output: [[A, 1], [A, 2], [B, 1], [B, 2]]
///
/// This is used to generate all shader permutations from the possible values
/// of each permutation parameter.
fn generate_cartesian_product(arrays: &[&Vec<toml::Value>]) -> Vec<Vec<toml::Value>> {
    // Base case: if no arrays provided, return a single empty combination
    if arrays.is_empty() {
        return vec![vec![]];
    }

    // Start with a single empty combination that we'll build upon
    // This represents "no choices made yet"
    let mut result = vec![vec![]];

    // Process each array one at a time
    // For each array, we take all existing combinations and extend them
    // with each possible value from the current array
    for array in arrays {
        let mut new_result = Vec::new();

        // For each existing combination we've built so far...
        for combination in &result {
            // ...try adding each value from the current array
            for value in *array {
                // Clone the existing combination and append the new value
                let mut new_combination = combination.clone();
                new_combination.push(value.clone());
                new_result.push(new_combination);
            }
        }

        // Replace the old combinations with the newly extended ones
        result = new_result;
    }

    // Example walkthrough with [[A, B], [1, 2]]:
    // Initial: result = [[]]
    //
    // After processing [A, B]:
    //   For combination []:
    //     Add A -> [A]
    //     Add B -> [B]
    //   result = [[A], [B]]
    //
    // After processing [1, 2]:
    //   For combination [A]:
    //     Add 1 -> [A, 1]
    //     Add 2 -> [A, 2]
    //   For combination [B]:
    //     Add 1 -> [B, 1]
    //     Add 2 -> [B, 2]
    //   result = [[A, 1], [A, 2], [B, 1], [B, 2]]

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
    cmd.arg(format!("-I{INCLUDE_DIR}"));

    // Add defines for this permutation
    for (key, value) in &permutation.defines {
        cmd.arg(format!("-D{key}={value}"));
    }

    // Set shader stage
    cmd.arg("-fshader-stage=comp");

    // Optimize
    cmd.arg("-O");

    // Input and output files (compile to memory first)
    cmd.arg(&permutation.shader_file);
    cmd.arg("-o");
    cmd.arg("-"); // Output to stdout

    // Execute the command
    let output = cmd.output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let command_str = format_command(&cmd);
        return Err(anyhow::anyhow!(
            "Shader compilation failed for permutation '{}'\n\
            Command: {}\n\
            Error: {}\n\
            \n\
            This usually indicates:\n\
            - Missing glslc compiler (install Vulkan SDK)\n\
            - Invalid shader syntax or defines\n\
            - Incorrect include paths",
            permutation.permutation_id,
            command_str,
            stderr
        ));
    }

    // Calculate hash of the compiled output
    let mut hasher = DefaultHasher::new();
    output.stdout.hash(&mut hasher);
    let hash = hasher.finish();
    let hash_hex = format!("{hash:016x}");

    // Get the base shader name (without extension)
    let shader_name = permutation.shader_file.file_stem().unwrap();

    // Create output filename based on shader name and hash (8 hex chars for readability)
    let output_path = permutation.output_directory.join(format!(
        "{}_{}.spv",
        shader_name,
        &hash_hex[..HASH_TRUNCATE_LEN]
    ));

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
    let mut hash_groups: IndexMap<String, Vec<&CompilationResult>> = IndexMap::new();

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
        // Count extra copies beyond the first one
        for &result in group.iter().skip(1) {
            total_duplicate_bytes += result.size_bytes;
        }
    }

    let space_saved_mb = total_duplicate_bytes as f64 / (1024.0 * 1024.0);

    DeduplicationInfo {
        unique_hashes,
        duplicates_eliminated,
        space_saved_mb,
    }
}

fn generate_rust_embedding(
    shader_config: &ShaderConfig,
    all_permutations: &[ShaderPermutation],
    compilation_results: &[CompilationResult],
) -> Result<()> {
    // Filter permutations and results for this specific shader configuration
    let relevant_permutations: Vec<_> = all_permutations
        .iter()
        .filter(|p| p.output_directory == shader_config.output_directory)
        .collect();

    let relevant_results: Vec<_> = compilation_results
        .iter()
        .filter(|r| r.output_path.starts_with(&shader_config.output_directory))
        .collect();

    if relevant_permutations.is_empty() || relevant_results.is_empty() {
        return Ok(()); // Nothing to generate for this set
    }

    println!(
        "Generating Rust embedding for {}",
        shader_config.output_directory
    );

    // Generate the Rust code
    let rust_code = generate_rust_code(shader_config, &relevant_permutations, &relevant_results)?;

    // Write to a .rs file in the same directory
    let rust_file_path = shader_config.output_directory.join(GENERATED_FILE_NAME);
    fs::write(&rust_file_path, rust_code)?;

    println!("Generated {rust_file_path}");

    Ok(())
}

fn generate_rust_code(
    shader_config: &ShaderConfig,
    permutations: &[&ShaderPermutation],
    results: &[&CompilationResult],
) -> Result<String> {
    let mut code = String::new();

    // Generate file header
    code.push_str("// Auto-generated by xtask compile-shaders\n");
    code.push_str("// DO NOT EDIT MANUALLY\n\n");

    // Generate enums for each permutation parameter
    generate_permutation_enums(&mut code, &shader_config.config)?;

    // Generate embedded shader data
    generate_embedded_shaders(&mut code, shader_config, results)?;

    // Generate shader struct
    generate_shader_struct(&mut code, shader_config, results)?;

    // Generate choice function
    generate_choice_function(&mut code, shader_config, permutations, results)?;

    Ok(code)
}

fn generate_permutation_enums(code: &mut String, config: &ShaderPermutationConfig) -> Result<()> {
    for (param_name, values) in &config.permutations {
        let enum_name = shader_param_to_enum_name(param_name);

        code.push_str("#[derive(Debug, Clone, Copy, PartialEq, Eq)]\n");
        code.push_str(&format!("pub enum {enum_name} {{\n"));

        // Check if values are exactly [0, 1] or [0] for binary enum
        let values_as_strings: Vec<String> = values.iter().map(toml_value_to_string).collect();
        let is_zero_one = values_as_strings == ["0".to_string(), "1".to_string()];
        let is_zero = values_as_strings == ["0".to_string()];

        if is_zero_one {
            code.push_str("    Off,\n");
            code.push_str("    On,\n");
        } else if is_zero {
            code.push_str("    Off,\n");
        } else {
            // More than 2 values, generate indexed variants
            for (i, _) in values.iter().enumerate() {
                let variant_name = format!("Value{i}");
                code.push_str(&format!("    {variant_name},\n"));
            }
        }

        code.push_str("}\n\n");
    }

    Ok(())
}

fn generate_embedded_shaders(
    code: &mut String,
    _shader_config: &ShaderConfig,
    results: &[&CompilationResult],
) -> Result<()> {
    // Get unique shaders by hash
    let mut unique_shaders: IndexMap<String, &CompilationResult> = IndexMap::new();
    for result in results {
        unique_shaders.insert(result.content_hash.clone(), *result);
    }

    code.push_str("
#[repr(align(4))]
struct Align4<const N: usize>([u8; N]);

macro_rules! include_shaders {
    () => {};
    ($NAME:ident = $PATH:expr, $($REST:tt)*) => {
        static $NAME: &'static [u8] = &(Align4::<{include_bytes!($PATH).len()}>(*include_bytes!($PATH)).0);
        include_shaders!{$($REST)*}
    };
    ($NAME:ident = $PATH:expr) => {
        include_shaders!{static $NAME = $path;};
    };
}

include_shaders! {
");

    for (hash, result) in &unique_shaders {
        let var_name = format!("SHADER_{}", &hash[..HASH_TRUNCATE_LEN].to_uppercase());

        code.push_str(&format!(
            "    {var_name} = \"{}\",\n",
            result.output_path.file_name().unwrap()
        ));
    }

    code.push_str("}\n\n");
    Ok(())
}

fn generate_shader_struct(
    code: &mut String,
    _shader_config: &ShaderConfig,
    results: &[&CompilationResult],
) -> Result<()> {
    // Get unique shader names
    let mut shader_names: IndexSet<String> = IndexSet::new();
    for result in results {
        if let Some(name) = result.output_path.file_stem() {
            // Extract shader name part (before the hash)
            if let Some(underscore_pos) = name.rfind('_') {
                let shader_name = &name[..underscore_pos];
                shader_names.insert(shader_name.to_string());
            }
        }
    }

    let mut shader_names: Vec<_> = shader_names.into_iter().collect();
    shader_names.sort();

    code.push_str("#[derive(Debug, Clone)]\n");
    code.push_str("pub struct Shaders {\n");

    for shader_name in &shader_names {
        // Convert shader name to field name
        let field_name = shader_name
            .replace("ffx_fsr3upscaler_", "")
            .replace("_pass", "");
        code.push_str(&format!("    pub {field_name}: &'static [u8],\n"));
    }

    code.push_str("}\n\n");
    Ok(())
}

/// Extract unique shader names from compilation results  
fn extract_shader_names(results: &[&CompilationResult]) -> Vec<String> {
    let mut shader_names: IndexSet<String> = IndexSet::new();
    for result in results {
        if let Some(name) = result.output_path.file_stem() {
            // Extract shader name part (before the hash)
            if let Some(underscore_pos) = name.rfind('_') {
                let shader_name = &name[..underscore_pos];
                shader_names.insert(shader_name.to_string());
            }
        }
    }
    let mut sorted_names: Vec<_> = shader_names.into_iter().collect();
    sorted_names.sort();
    sorted_names
}

/// Generate function signature for shader selection function
fn generate_choice_function_signature(code: &mut String, param_info: &[ParamInfo]) {
    code.push_str("#[inline(always)]\n");
    code.push_str("pub fn choose_shaders(");
    for (i, param) in param_info.iter().enumerate() {
        if i > 0 {
            code.push_str(", ");
        }
        code.push_str(&format!("{}: {}", param.name_lower, param.enum_name));
    }
    code.push_str(") -> Shaders {\n");
}

/// Generate match patterns and shader assignments for choice function
fn generate_choice_function_matches(
    code: &mut String,
    param_info: &[ParamInfo],
    result_lookup: &IndexMap<String, &CompilationResult>,
    shader_names: &[String],
) {
    code.push_str("    match (");
    for (i, param) in param_info.iter().enumerate() {
        if i > 0 {
            code.push_str(", ");
        }
        code.push_str(&param.name_lower);
    }
    code.push_str(") {\n");

    // Generate all possible enum combinations and map them to shaders
    let combinations = generate_all_enum_combinations(param_info);

    for combination in combinations {
        // Convert enum combination to TOML values for lookup
        let lookup_values: Vec<String> = combination
            .iter()
            .map(|enum_variant| match enum_variant.as_str() {
                "On" => "1".to_string(),
                "Off" => "0".to_string(),
                _ => "0".to_string(), // Default case
            })
            .collect();

        // Generate match pattern
        code.push_str("        (");
        for (i, enum_variant) in combination.iter().enumerate() {
            if i > 0 {
                code.push_str(", ");
            }
            let ParamInfo { enum_name, .. } = &param_info[i];
            code.push_str(&format!("{enum_name}::{enum_variant}"));
        }
        code.push_str(") => Shaders {\n");

        // Generate shader assignments for this permutation
        for shader_name in shader_names {
            let field_name = shader_name
                .replace("ffx_fsr3upscaler_", "")
                .replace("_pass", "");

            // Build permutation ID to look up the result
            let mut permutation_id = shader_name.clone();
            for value in &lookup_values {
                permutation_id.push('_');
                permutation_id.push_str(value);
            }

            if let Some(result) = result_lookup.get(&permutation_id) {
                let hash_short = &result.content_hash[..HASH_TRUNCATE_LEN].to_uppercase();
                code.push_str(&format!("            {field_name}: SHADER_{hash_short},\n"));
            } else {
                // This shouldn't happen, but provide a fallback
                code.push_str(&format!("            {field_name}: &[],\n"));
            }
        }

        code.push_str("        },\n");
    }

    code.push_str("    }\n");
    code.push_str("}\n");
}

struct ParamInfo {
    name_lower: String,
    enum_name: String,
    values: Vec<String>,
}

fn generate_choice_function(
    code: &mut String,
    shader_config: &ShaderConfig,
    _permutations: &[&ShaderPermutation],
    results: &[&CompilationResult],
) -> Result<()> {
    // Build result lookup by permutation ID
    let result_lookup: IndexMap<String, &CompilationResult> = results
        .iter()
        .map(|r| (r.permutation_id.clone(), *r))
        .collect();

    // Get parameter info for function signature
    let mut param_info = Vec::with_capacity(shader_config.config.permutations.len());
    for (param_name, values) in &shader_config.config.permutations {
        let enum_name = shader_param_to_enum_name(param_name);
        let param_name_lower = shader_param_to_function_param(param_name);
        let value_strings: Vec<String> = values.iter().map(toml_value_to_string).collect();
        param_info.push(ParamInfo {
            name_lower: param_name_lower,
            enum_name,
            values: value_strings,
        });
    }

    // Get all unique shader names from results
    let shader_names = extract_shader_names(results);

    // Generate function signature
    generate_choice_function_signature(code, &param_info);

    // Generate match patterns and shader assignments
    generate_choice_function_matches(code, &param_info, &result_lookup, &shader_names);

    Ok(())
}

fn generate_all_enum_combinations(param_info: &[ParamInfo]) -> Vec<Vec<String>> {
    // For binary enums, generate all On/Off combinations
    let mut combinations = vec![vec![]];

    for param in param_info {
        let mut new_combinations = Vec::new();
        for combination in &combinations {
            for variant in &param.values {
                let variant_translation = match variant.as_str() {
                    "1" => "On",
                    "0" => "Off",
                    v => v,
                };
                let mut new_combination = combination.clone();
                new_combination.push(variant_translation.to_string());
                new_combinations.push(new_combination);
            }
        }
        combinations = new_combinations;
    }

    combinations
}
