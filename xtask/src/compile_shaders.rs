//! Compiles FidelityFX GLSL shaders to SPIR-V with permutation support,
//! deduplication, and Rust code generation.

use anyhow::Result;
use camino::{Utf8Path, Utf8PathBuf};
use indexmap::IndexMap;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Deserialize;
use std::{
    collections::{BTreeSet, hash_map::DefaultHasher},
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

const SDK_BASE_PATH: &str = "shaders/src";
const INCLUDE_DIR: &str = "shaders/include";
const SHADERS_DIR: &str = "wgpu-ffx-shaders-spv/src";
const PERM_CONFIG_FILE: &str = "perm.toml";
const GENERATED_FILE_NAME: &str = "mod.rs";
const HASH_TRUNCATE_LEN: usize = 8;

#[derive(Debug, Deserialize)]
struct ShaderPathConfig {
    subdirectory: String,
    #[serde(default)]
    prefix: Option<String>,
    #[serde(default)]
    suffix: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FieldPermutationConfig {
    values: Vec<toml::Value>,
    /// Suffix appended to struct field names for non-default variants.
    suffix: String,
}

/// One `[permutations]` entry: either a plain value list whose variant names
/// are derived (`[0, 1]` -> Off/On), or explicit variant-name -> define-value
/// pairs (`{ Core = 0, Tier2 = 1, Native = 2 }`) in declaration order.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PermutationSpec {
    Values(Vec<toml::Value>),
    Named(IndexMap<String, toml::Value>),
}

impl PermutationSpec {
    fn values(&self) -> Vec<&toml::Value> {
        match self {
            PermutationSpec::Values(values) => values.iter().collect(),
            PermutationSpec::Named(map) => map.values().collect(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ShaderPermutationConfig {
    path: ShaderPathConfig,
    base: IndexMap<String, toml::Value>,
    permutations: IndexMap<String, PermutationSpec>,
    #[serde(default)]
    field_permutations: IndexMap<String, FieldPermutationConfig>,
}

#[derive(Debug)]
struct ShaderConfig {
    config: ShaderPermutationConfig,
    sdk_shader_directory: Utf8PathBuf,
    output_directory: Utf8PathBuf,
}

#[derive(Debug, Clone)]
struct PermVariant {
    define_value: String,
    variant_name: String,
}

#[derive(Debug)]
struct PermutationParam {
    enum_name: String,
    param_name: String,
    variants: Vec<PermVariant>,
}

#[derive(Debug, Clone)]
struct FieldPermVariant {
    define_value: String,
    /// `None` for the default (first) variant.
    field_suffix: Option<String>,
}

#[derive(Debug)]
struct FieldPermParam {
    variants: Vec<FieldPermVariant>,
}

#[derive(Debug, Clone)]
struct ShaderPermutation {
    shader_file: Utf8PathBuf,
    permutation_id: String,
    defines: Vec<(String, String)>,
    output_directory: Utf8PathBuf,
}

#[derive(Debug, Clone)]
struct CompilationResult {
    shader_name: String,
    permutation_id: String,
    content_hash: String,
    output_path: Utf8PathBuf,
    size_bytes: usize,
}

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

    let remaining = args.finish();
    if !remaining.is_empty() {
        return Err(anyhow::anyhow!("Unexpected arguments: {remaining:?}"));
    }

    println!("Compiling shaders...");

    let shader_configs = discover_shader_configs()?;

    if shader_configs.is_empty() {
        println!("No perm.toml files found in shaders/ directory");
        return Ok(());
    }

    println!("Found {} shader configurations", shader_configs.len());

    clean_old_spv_files(&shader_configs)?;

    let (all_permutations, total_glsl_files) = generate_all_shader_permutations(&shader_configs)?;

    println!("Found {total_glsl_files} GLSL files across all configurations");

    let (successful_results, failure_count) = compile_all_permutations(&all_permutations);

    let dedup_info = analyze_deduplication(&successful_results);

    println!("Compilation and deduplication complete!");
    println!("Original permutations: {}", all_permutations.len());
    println!("Successful compilations: {}", successful_results.len());
    if failure_count > 0 {
        println!("Failed compilations: {failure_count}");
    }
    println!("Unique shader variants: {}", dedup_info.unique_hashes);
    println!(
        "Duplicates eliminated: {}",
        dedup_info.duplicates_eliminated
    );
    println!("Space saved: {:.2} MB", dedup_info.space_saved_mb);

    for shader_config in &shader_configs {
        generate_rust_embedding(shader_config, &successful_results)?;
    }

    if failure_count > 0 {
        return Err(anyhow::anyhow!(
            "{failure_count} shader permutations failed to compile"
        ));
    }

    Ok(())
}

fn load_shader_config(config_path: &Utf8Path) -> Result<ShaderPermutationConfig> {
    let content = fs::read_to_string(config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read shader config {config_path}: {e}"))?;

    let config: ShaderPermutationConfig = toml::from_str(&content)
        .map_err(|e| anyhow::anyhow!("Failed to parse shader config {config_path}: {e}"))?;

    for (define_name, spec) in &config.permutations {
        if let PermutationSpec::Named(map) = spec {
            for name in map.keys() {
                if !is_valid_variant_name(name) {
                    return Err(anyhow::anyhow!(
                        "{config_path}: variant name `{name}` of {define_name} \
                        is not a valid Rust identifier"
                    ));
                }
            }
        }
        let values: Vec<String> = spec
            .values()
            .into_iter()
            .map(toml_value_to_define_string)
            .collect();
        let unique: BTreeSet<&str> = values.iter().map(String::as_str).collect();
        if unique.len() != values.len() {
            return Err(anyhow::anyhow!(
                "{config_path}: duplicate values in permutation {define_name}: {values:?}"
            ));
        }
    }

    Ok(config)
}

/// Valid as a Rust enum variant: an XID-start-ish ASCII identifier.
fn is_valid_variant_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
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

    for entry in fs::read_dir(shaders_dir)? {
        let entry = entry?;
        let path = Utf8PathBuf::try_from(entry.path())
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in shader directory path: {e}"))?;

        if path.is_dir() {
            let perm_file = path.join(PERM_CONFIG_FILE);
            if perm_file.exists() {
                let config = load_shader_config(&perm_file)?;
                let sdk_shader_directory =
                    Utf8PathBuf::from(SDK_BASE_PATH).join(&config.path.subdirectory);

                shader_configs.push(ShaderConfig {
                    config,
                    sdk_shader_directory,
                    output_directory: path,
                });
            }
        }
    }

    Ok(shader_configs)
}

fn clean_old_spv_files(shader_configs: &[ShaderConfig]) -> Result<()> {
    println!("Cleaning up old .spv files...");
    let mut total_removed = 0;

    for shader_config in shader_configs {
        let output_dir = &shader_config.output_directory;

        if !output_dir.exists() {
            continue;
        }

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

        let permutations = generate_permutations_for_config(
            &glsl_files,
            &shader_config.config,
            &shader_config.output_directory,
        );
        all_permutations.extend(permutations);
    }

    Ok((all_permutations, total_glsl_files))
}

fn generate_permutations_for_config(
    glsl_files: &[Utf8PathBuf],
    config: &ShaderPermutationConfig,
    output_directory: &Utf8Path,
) -> Vec<ShaderPermutation> {
    // Both regular and field permutations participate in the cartesian product.
    let mut all_define_names: Vec<&str> = Vec::new();
    let mut all_value_arrays: Vec<Vec<&toml::Value>> = Vec::new();

    for (name, spec) in &config.permutations {
        all_define_names.push(name);
        all_value_arrays.push(spec.values());
    }
    for (name, fc) in &config.field_permutations {
        all_define_names.push(name);
        all_value_arrays.push(fc.values.iter().collect());
    }

    let sizes: Vec<usize> = all_value_arrays.iter().map(|v| v.len()).collect();
    let index_combos = cartesian_product_indices(&sizes);

    let base_defines: Vec<(String, String)> = config
        .base
        .iter()
        .map(|(k, v)| (k.clone(), toml_value_to_define_string(v)))
        .collect();

    let mut permutations = Vec::new();

    for shader_file in glsl_files {
        let shader_stem = shader_file.file_stem().unwrap();

        for indices in &index_combos {
            let mut defines = base_defines.clone();
            let mut permutation_id = shader_stem.to_string();

            for (i, &idx) in indices.iter().enumerate() {
                let value_str = toml_value_to_define_string(all_value_arrays[i][idx]);
                permutation_id.push('_');
                permutation_id.push_str(&value_str);
                defines.push((all_define_names[i].to_string(), value_str));
            }

            permutations.push(ShaderPermutation {
                shader_file: shader_file.clone(),
                permutation_id,
                defines,
                output_directory: output_directory.to_path_buf(),
            });
        }
    }

    permutations
}

fn compile_all_permutations(
    all_permutations: &[ShaderPermutation],
) -> (Vec<CompilationResult>, usize) {
    println!("Generating {} total permutations", all_permutations.len());

    let progress = ProgressBar::new(all_permutations.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let results: Vec<Option<CompilationResult>> = all_permutations
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

    let failure_count = results.iter().filter(|r| r.is_none()).count();
    let successes = results.into_iter().flatten().collect();
    (successes, failure_count)
}

fn compile_single_permutation(permutation: &ShaderPermutation) -> Result<CompilationResult> {
    let mut cmd = Command::new("glslc");
    cmd.arg("--target-env=vulkan1.2");
    cmd.arg(format!("-I{INCLUDE_DIR}"));

    for (key, value) in &permutation.defines {
        cmd.arg(format!("-D{key}={value}"));
    }

    cmd.arg("-fshader-stage=comp");
    cmd.arg("-O");
    cmd.arg(&permutation.shader_file);
    cmd.arg("-o");
    cmd.arg("-");

    let output = cmd.output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!(
            "Shader compilation failed for permutation '{}'\n\
            Command: {:?}\n\
            Error: {}\n\
            \n\
            This usually indicates:\n\
            - Missing glslc compiler (install Vulkan SDK)\n\
            - Invalid shader syntax or defines\n\
            - Incorrect include paths",
            permutation.permutation_id,
            cmd,
            stderr
        ));
    }

    let mut hasher = DefaultHasher::new();
    output.stdout.hash(&mut hasher);
    let hash_hex = format!("{:016x}", hasher.finish());

    let shader_name = permutation.shader_file.file_stem().unwrap().to_string();

    let output_path = permutation.output_directory.join(format!(
        "{}_{}.spv",
        shader_name,
        &hash_hex[..HASH_TRUNCATE_LEN]
    ));

    fs::create_dir_all(output_path.parent().unwrap())?;
    fs::write(&output_path, &output.stdout)?;

    Ok(CompilationResult {
        shader_name,
        permutation_id: permutation.permutation_id.clone(),
        content_hash: hash_hex,
        output_path,
        size_bytes: output.stdout.len(),
    })
}

fn analyze_deduplication(results: &[CompilationResult]) -> DeduplicationInfo {
    let mut hash_groups: IndexMap<&str, Vec<&CompilationResult>> = IndexMap::new();

    for result in results {
        hash_groups
            .entry(&result.content_hash)
            .or_default()
            .push(result);
    }

    let unique_hashes = hash_groups.len();
    let duplicates_eliminated = results.len() - unique_hashes;

    let total_duplicate_bytes: usize = hash_groups
        .values()
        .flat_map(|group| group.iter().skip(1))
        .map(|r| r.size_bytes)
        .sum();

    DeduplicationInfo {
        unique_hashes,
        duplicates_eliminated,
        space_saved_mb: total_duplicate_bytes as f64 / (1024.0 * 1024.0),
    }
}

fn generate_rust_embedding(
    shader_config: &ShaderConfig,
    compilation_results: &[CompilationResult],
) -> Result<()> {
    let relevant_results: Vec<_> = compilation_results
        .iter()
        .filter(|r| r.output_path.starts_with(&shader_config.output_directory))
        .collect();

    if relevant_results.is_empty() {
        return Ok(());
    }

    println!(
        "Generating Rust embedding for {}",
        shader_config.output_directory
    );

    let rust_code = generate_rust_code(shader_config, &relevant_results)?;
    let rust_file_path = shader_config.output_directory.join(GENERATED_FILE_NAME);
    fs::write(&rust_file_path, rust_code)?;

    println!("Generated {rust_file_path}");
    Ok(())
}

fn generate_rust_code(
    shader_config: &ShaderConfig,
    results: &[&CompilationResult],
) -> Result<String> {
    let params = build_permutation_params(&shader_config.config);
    let field_params = build_field_perm_params(&shader_config.config);
    let shader_names = unique_shader_names(results);

    let mut code = String::new();
    code.push_str("// Auto-generated by xtask compile-shaders\n");
    code.push_str("// DO NOT EDIT MANUALLY\n\n");

    // Permutation enums
    for param in &params {
        code.push_str("#[derive(Debug, Clone, Copy, PartialEq, Eq)]\n");
        code.push_str(&format!("pub enum {} {{\n", param.enum_name));
        for variant in &param.variants {
            code.push_str(&format!("    {},\n", variant.variant_name));
        }
        code.push_str("}\n\n");
    }

    // Embedded shader data (deduplicated by content hash)
    let mut unique_shaders: IndexMap<&str, &CompilationResult> = IndexMap::new();
    for result in results {
        unique_shaders.entry(&result.content_hash).or_insert(result);
    }

    code.push_str(
        "\n\
        #[repr(align(4))]\n\
        struct Align4<const N: usize>([u8; N]);\n\
        \n\
        macro_rules! include_shaders {\n    \
            () => {};\n    \
            ($NAME:ident = $PATH:expr, $($REST:tt)*) => {\n        \
                static $NAME: &'static [u8] = &(Align4::<{include_bytes!($PATH).len()}>(*include_bytes!($PATH)).0);\n        \
                include_shaders!{$($REST)*}\n    \
            };\n\
        }\n\
        \n\
        include_shaders! {\n",
    );

    for (&hash, result) in &unique_shaders {
        let var_name = format!("SHADER_{}", hash[..HASH_TRUNCATE_LEN].to_uppercase());
        code.push_str(&format!(
            "    {var_name} = \"{}\",\n",
            result.output_path.file_name().unwrap()
        ));
    }

    code.push_str("}\n\n");

    // Shader struct
    let suffix_combos = field_suffix_combinations(&field_params);

    code.push_str("#[derive(Debug, Clone)]\n");
    code.push_str("pub struct Shaders {\n");

    for shader_name in &shader_names {
        let base_field = derive_field_name(shader_name, &shader_config.config.path);
        for suffixes in &suffix_combos {
            let field_name = apply_field_suffixes(&base_field, suffixes);
            code.push_str(&format!("    pub {field_name}: &'static [u8],\n"));
        }
    }

    code.push_str("}\n\n");

    // Choice function
    let result_lookup: IndexMap<&str, &CompilationResult> = results
        .iter()
        .map(|r| (r.permutation_id.as_str(), *r))
        .collect();

    let field_sizes: Vec<usize> = field_params.iter().map(|fp| fp.variants.len()).collect();
    let field_index_combos = cartesian_product_indices(&field_sizes);

    code.push_str("#[inline(always)]\n");
    code.push_str("pub fn choose_shaders(");
    for (i, param) in params.iter().enumerate() {
        if i > 0 {
            code.push_str(", ");
        }
        code.push_str(&format!("{}: {}", param.param_name, param.enum_name));
    }
    code.push_str(") -> Shaders {\n");

    code.push_str("    match (");
    for (i, param) in params.iter().enumerate() {
        if i > 0 {
            code.push_str(", ");
        }
        code.push_str(&param.param_name);
    }
    code.push_str(") {\n");

    let param_sizes: Vec<usize> = params.iter().map(|p| p.variants.len()).collect();
    let regular_combos = cartesian_product_indices(&param_sizes);

    for regular_indices in &regular_combos {
        code.push_str("        (");
        for (i, &idx) in regular_indices.iter().enumerate() {
            if i > 0 {
                code.push_str(", ");
            }
            let param = &params[i];
            code.push_str(&format!(
                "{}::{}",
                param.enum_name, param.variants[idx].variant_name
            ));
        }
        code.push_str(") => Shaders {\n");

        for shader_name in &shader_names {
            let base_field = derive_field_name(shader_name, &shader_config.config.path);

            for field_indices in &field_index_combos {
                let mut permutation_id = shader_name.clone();
                for (i, &idx) in regular_indices.iter().enumerate() {
                    permutation_id.push('_');
                    permutation_id.push_str(&params[i].variants[idx].define_value);
                }
                for (i, &idx) in field_indices.iter().enumerate() {
                    permutation_id.push('_');
                    permutation_id.push_str(&field_params[i].variants[idx].define_value);
                }

                let suffixes: Vec<Option<&str>> = field_indices
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| field_params[i].variants[idx].field_suffix.as_deref())
                    .collect();
                let field_name = apply_field_suffixes(&base_field, &suffixes);

                if let Some(result) = result_lookup.get(permutation_id.as_str()) {
                    let hash_var = format!(
                        "SHADER_{}",
                        &result.content_hash[..HASH_TRUNCATE_LEN].to_uppercase()
                    );
                    code.push_str(&format!("            {field_name}: {hash_var},\n"));
                } else {
                    code.push_str(&format!("            {field_name}: &[],\n"));
                }
            }
        }

        code.push_str("        },\n");
    }

    code.push_str("    }\n");
    code.push_str("}\n");

    Ok(code)
}

fn build_permutation_params(config: &ShaderPermutationConfig) -> Vec<PermutationParam> {
    config
        .permutations
        .iter()
        .map(|(define_name, spec)| {
            let variants = match spec {
                PermutationSpec::Values(values) => {
                    let variant_names = variant_names_for_values(values);
                    values
                        .iter()
                        .zip(variant_names)
                        .map(|(v, name)| PermVariant {
                            define_value: toml_value_to_define_string(v),
                            variant_name: name,
                        })
                        .collect()
                }
                PermutationSpec::Named(map) => map
                    .iter()
                    .map(|(name, v)| PermVariant {
                        define_value: toml_value_to_define_string(v),
                        variant_name: name.clone(),
                    })
                    .collect(),
            };

            PermutationParam {
                enum_name: define_to_enum_name(define_name),
                param_name: define_to_param_name(define_name),
                variants,
            }
        })
        .collect()
}

fn build_field_perm_params(config: &ShaderPermutationConfig) -> Vec<FieldPermParam> {
    config
        .field_permutations
        .iter()
        .map(|(_, fc)| {
            let variants = fc
                .values
                .iter()
                .enumerate()
                .map(|(i, v)| FieldPermVariant {
                    define_value: toml_value_to_define_string(v),
                    field_suffix: if i == 0 {
                        None
                    } else {
                        Some(fc.suffix.clone())
                    },
                })
                .collect();

            FieldPermParam { variants }
        })
        .collect()
}

fn toml_value_to_define_string(value: &toml::Value) -> String {
    match value {
        toml::Value::Integer(i) => i.to_string(),
        toml::Value::String(s) => s.clone(),
        toml::Value::Boolean(b) => if *b { "1" } else { "0" }.to_string(),
        toml::Value::Float(f) => f.to_string(),
        _ => value.to_string(),
    }
}

/// `[0, 1]` -> `["Off", "On"]`, `[0]` -> `["Off"]`, otherwise `["Value0", "Value1", ...]`.
///
/// Defines whose values are not a simple on/off toggle should use the
/// [`PermutationSpec::Named`] form instead of relying on the fallback names.
fn variant_names_for_values(values: &[toml::Value]) -> Vec<String> {
    let strs: Vec<String> = values.iter().map(toml_value_to_define_string).collect();
    if strs == ["0", "1"] {
        vec!["Off".to_string(), "On".to_string()]
    } else if strs == ["0"] {
        vec!["Off".to_string()]
    } else {
        (0..values.len()).map(|i| format!("Value{i}")).collect()
    }
}

/// `"FFX_HALF"` -> `"Half"`, `"FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT"` -> `"Fsr3upscalerOptionHdrColorInput"`.
fn define_to_enum_name(define_name: &str) -> String {
    define_name
        .strip_prefix("FFX_")
        .unwrap_or(define_name)
        .to_lowercase()
        .split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}

/// `"FFX_HALF"` -> `"half"`, `"FFX_FSR3UPSCALER_OPTION_HDR"` -> `"fsr3upscaler_option_hdr"`.
fn define_to_param_name(define_name: &str) -> String {
    define_name
        .strip_prefix("FFX_")
        .unwrap_or(define_name)
        .to_lowercase()
}

fn derive_field_name(shader_name: &str, path_config: &ShaderPathConfig) -> String {
    let mut name = shader_name;
    if let Some(prefix) = &path_config.prefix {
        name = name.strip_prefix(prefix.as_str()).unwrap_or(name);
    }
    if let Some(suffix) = &path_config.suffix {
        name = name.strip_suffix(suffix.as_str()).unwrap_or(name);
    }
    name.to_string()
}

fn unique_shader_names(results: &[&CompilationResult]) -> Vec<String> {
    results
        .iter()
        .map(|r| r.shader_name.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// For a single field perm with values `[0, 1]` and suffix `"sharpen"`:
/// returns `[[None], [Some("sharpen")]]`.
fn field_suffix_combinations(field_params: &[FieldPermParam]) -> Vec<Vec<Option<&str>>> {
    let sizes: Vec<usize> = field_params.iter().map(|fp| fp.variants.len()).collect();
    cartesian_product_indices(&sizes)
        .into_iter()
        .map(|indices| {
            indices
                .iter()
                .enumerate()
                .map(|(i, &idx)| field_params[i].variants[idx].field_suffix.as_deref())
                .collect()
        })
        .collect()
}

/// `"accumulate"` + `[Some("sharpen")]` -> `"accumulate_sharpen"`.
fn apply_field_suffixes(base: &str, suffixes: &[Option<&str>]) -> String {
    let mut name = base.to_string();
    for s in suffixes.iter().flatten() {
        name.push('_');
        name.push_str(s);
    }
    name
}

/// `cartesian_product_indices(&[2, 3])` -> `[[0,0], [0,1], [0,2], [1,0], [1,1], [1,2]]`
fn cartesian_product_indices(sizes: &[usize]) -> Vec<Vec<usize>> {
    let mut result = vec![vec![]];
    for &size in sizes {
        let mut next = Vec::with_capacity(result.len() * size);
        for combo in &result {
            for i in 0..size {
                let mut new_combo = combo.clone();
                new_combo.push(i);
                next.push(new_combo);
            }
        }
        result = next;
    }
    result
}
