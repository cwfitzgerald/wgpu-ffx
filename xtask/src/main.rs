use anyhow::Result;

mod compile_shaders;

const HELP: &str = r#"xtask

USAGE:
    xtask [OPTIONS] <SUBCOMMAND>

OPTIONS:
    -h, --help    Print help information

SUBCOMMANDS:
    compile-shaders    Compile shaders for the project
"#;

fn main() {
    if let Err(e) = try_main() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn try_main() -> Result<()> {
    let mut pargs = pico_args::Arguments::from_env();

    if pargs.contains(["-h", "--help"]) {
        print!("{HELP}");
        return Ok(());
    }

    let subcommand = pargs.subcommand()?.unwrap_or_default();

    match subcommand.as_str() {
        "compile-shaders" => compile_shaders::compile_shaders(pargs)?,
        cmd => {
            if cmd.is_empty() {
                println!("{HELP}");
            } else {
                eprintln!("Unknown subcommand: {cmd}");
                eprintln!();
                println!("{HELP}");
                return Err(anyhow::anyhow!("Unknown subcommand: {cmd}"));
            }
        }
    }

    Ok(())
}
