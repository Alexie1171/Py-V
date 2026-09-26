"""
language_detector.py — PY-V (inference/engine/)
Which programming language a message asks about, from its words (Phase 9
starts here), and the open file's language from VS Code's languageId (chat
panel file reading, Phase 10.2 — file_language()).
None = Python, V's own language — her tested templates and adapter.

Owner, 2026-09-26: any language at all, "from assembly to the latest" — so
layers, broad on purpose:
  1. KNOWN — ~150 names safe as bare words (haskell, cobol, zig, verilog, ...)
  2. SHORT — names that are letters / short words (go, c, d, r, swift, julia,
     assembly, ...): after "in / using / written in / convert to ..." or before
     "code / program / function / ..." ("in go", "c code", "a julia function")
  3. WORDY — names that are everyday words (basic, red, lean, move, scheme,
     ...): only next to "language / lang / compiler / interpreter / syntax"
  4. FORMATS — json, yaml, xml, html, css, ...: only when she is asked to
     make one ("write a yaml config"), not to read or convert one — "read a
     json file" is a Python task
  5. any other word in an unmistakable language phrase: "hello world in foo",
     "written in foo", "the foo language", "a language called foo"
The prompts (prompt_templates.OTHER_LANGUAGE_TEMPLATES) take any name, so a
language only has to be recognised here, not known. Words can't catch a
language shown only by pasted code — the open file's type covers the panel.

Several languages in one message: a target after "to" / "into" wins
("convert this python to typescript"), then Python if named — or one of its
libraries ("sqlite in python", "parse html with beautifulsoup") — then the
first other language.
"""

import re
from typing import Optional

_I = re.IGNORECASE

# 1 — display name: spellings (regex, matched as whole words)
KNOWN = {
    # assembly and low level
    "Assembly": r"assembly language|x86(?:-64)?(?: assembly| asm)?|x64 assembly|arm(?:64)? assembly|aarch64|"
                r"mips assembly|risc-?v(?: assembly)?|nasm|masm|yasm|6502 assembly|z80 assembly|avr assembly",
    "WebAssembly": r"webassembly|wasm", "LLVM IR": r"llvm ir",
    # C family and systems
    "C++": r"c\+\+|cpp|cplusplus", "C#": r"c#|c sharp|csharp", "Objective-C": r"objective-?c|obj-?c",
    "Rust": r"rust|rustlang", "Zig": r"zig|ziglang", "Nim": r"nim|nimlang", "Mojo": r"mojo", "Vlang": r"vlang",
    "Cython": r"cython", "CUDA": r"cuda", "OpenCL": r"opencl", "GLSL": r"glsl", "HLSL": r"hlsl", "WGSL": r"wgsl",
    # JVM and .NET
    "Java": r"java(?! ?script)", "Kotlin": r"kotlin", "Scala": r"scala", "Groovy": r"groovy|gradle script",
    "Clojure": r"clojure(?:script)?", "F#": r"f#|fsharp|f sharp", "Visual Basic": r"visual basic|vb\.net|vba|vbscript",
    # web and scripting
    "TypeScript": r"type ?script|tsx", "JavaScript": r"java ?script|node\.?js|nodejs|jsx|ecmascript|deno|react(?:js)?|"
                  r"vue(?:\.?js)?|angular|jquery|express\.?js|next\.?js",
    "CoffeeScript": r"coffee ?script", "PureScript": r"purescript", "ReScript": r"rescript",
    "PHP": r"php|laravel|symfony|wordpress", "Ruby": r"ruby|rails", "Perl": r"perl", "Raku": r"raku",
    "SCSS": r"scss|sass", "GraphQL": r"graphql", "LaTeX": r"latex",
    "Bash": r"bash|zsh|shell script|posix sh", "PowerShell": r"powershell|pwsh", "Batch": r"batch file|batch script|cmd script",
    "AWK": r"awk|gawk", "Lua": r"lua|luau", "Tcl": r"tcl", "AutoHotkey": r"autohotkey|ahk", "AppleScript": r"applescript",
    "Dart": r"flutter|dartlang", "Swift": r"swiftui", "Haxe": r"haxe",
    # functional and logic
    "Haskell": r"haskell", "OCaml": r"ocaml", "Erlang": r"erlang", "Elixir": r"elixir|phoenix liveview",
    "Lisp": r"lisp|common lisp|emacs lisp|elisp", "Prolog": r"prolog", "Idris": r"idris", "Agda": r"agda",
    "Coq": r"coq|rocq", "Lean": r"lean ?4|lean prover", "Isabelle": r"isabelle/hol", "Smalltalk": r"smalltalk|pharo",
    "APL": r"apl", "Brainfuck": r"brainf\*+k|brainfuck", "Befunge": r"befunge",
    # data, science, business
    "SQL": r"sql|mysql|postgres(?:ql)?|pl/sql|plsql|pl/pgsql|t-sql|tsql|mariadb|bigquery|snowflake",
    "R": r"rlang|tidyverse|ggplot2?|rstudio", "Julia": r"julialang", "MATLAB": r"matlab|simulink",
    "SAS": r"sas code|sas program", "Stata": r"stata", "Wolfram Language": r"wolfram language|mathematica",
    "COBOL": r"cobol", "Fortran": r"fortran", "ABAP": r"abap", "Apex": r"salesforce apex|apex class",
    "RPG": r"rpgle", "PL/I": r"pl/i", "Delphi": r"delphi", "Pascal": r"object pascal|free ?pascal",
    "Modula-2": r"modula-?2", "Oberon": r"oberon", "Simula": r"simula", "ALGOL": r"algol",
    "BASIC": r"qbasic|quickbasic|freebasic", "Ada": r"spark ada",
    # hardware, blockchain, infrastructure
    "Verilog": r"verilog", "SystemVerilog": r"systemverilog", "VHDL": r"vhdl", "Chisel": r"chisel hdl",
    "Solidity": r"solidity", "Vyper": r"vyper", "Makefile": r"makefile", "CMake": r"cmake",
    "Dockerfile": r"dockerfile", "Terraform": r"terraform|hcl", "Nix": r"nixos|nix flake|nix expression",
}

# 1b — frameworks, platforms and tools that mean a language
FRAMEWORKS = {
    "C++": r"arduino(?: sketch)?|unreal engine|ue[45] (?:c\+\+|actor|plugin)|qt (?:widgets?|window|app|application|creator|quick|gui|signals?)|win32 api|opengl app",
    "C#": r"unity (?:script|game|engine|project|component|editor|monobehaviou?r)|in unity|asp\.net(?: core)?|\.net(?: core)?|dotnet|blazor|wpf|winforms|xamarin|maui",
    "GDScript": r"godot|gdscript",
    "Java": r"spring (?:boot|framework|mvc|bean|controller|app|application|security|data)|jakarta ee|javafx|minecraft plugin",
    "Kotlin": r"android (?:app|activity|fragment|studio|service|view|intent)|jetpack compose|ktor",
    "Swift": r"ios app|uikit|xcode project",
    "JavaScript": r"svelte(?:kit)?|electron app|react native|expo app|ember\.?js|nestjs|nuxt|astro component|"
                  r"chrome extension|browser extension|google apps script",
    "C": r"linux kernel module|kernel module|embedded c|win32 c|posix threads|pthreads",
    "GLSL": r"shader|shadertoy|fragment shader|vertex shader",
    "Excel formula": r"excel formula|spreadsheet formula|google sheets formula|sheets formula|xlookup|vlookup",
    "Groovy": r"jenkins ?file|jenkins pipeline",
    "YAML": r"ansible(?: playbook)?|kubernetes(?: manifest| deployment| yaml)?|k8s|helm chart|github actions?|"
            r"gitlab ci|docker[- ]compose|circleci",
}

# 2 — letters / short words: after "in X" (and friends) or before a code noun
SHORT = {
    "Go": r"go|golang", "C": r"c", "D": r"d", "R": r"r", "Swift": r"swift", "Dart": r"dart", "Julia": r"julia",
    "Elm": r"elm", "Racket": r"racket", "Zig": r"zig", "Assembly": r"assembly|asm", "Pascal": r"pascal",
    "Ada": r"ada", "Gleam": r"gleam", "Odin": r"odin", "Chapel": r"chapel", "Pony": r"pony", "Crystal": r"crystal",
    "Octave": r"octave", "Carbon": r"carbon", "Hare": r"hare",
}
_BEFORE = (r"(?:in|using|written in|coded in|implemented in|into|"
           r"(?:convert|port|translate|rewrite|turn)\w*\b[^.?!\n]{0,60}?\b(?:to|into|in))")
_AFTER = (r"(?:code|program|programs|programming|language|lang|script|syntax|function|functions|example|snippet|"
          r"compiler|interpreter|file|files|module|struct|macro|project|library|package|crate|developer|hello,? world)")

# 3 — everyday words: only next to a word that makes them a language
WORDY = {
    "BASIC": r"basic", "Red": r"red", "Lean": r"lean", "Move": r"move", "Scheme": r"scheme", "Logo": r"logo",
    "Forth": r"forth", "Factor": r"factor", "Mercury": r"mercury", "Hack": r"hack", "Cairo": r"cairo",
    "Scratch": r"scratch", "Eiffel": r"eiffel", "Io": r"io", "Ballerina": r"ballerina", "Q": r"q", "J": r"j", "K": r"k",
}
_LANGUAGE_WORD = r"(?:programming language|language|lang|compiler|interpreter|syntax)"

# 4 — formats: only when she is asked to make one, never to read / convert one
FORMATS = {
    "JSON": r"json", "YAML": r"yaml|yml", "XML": r"xml|xslt", "TOML": r"toml", "Markdown": r"markdown",
    "HTML": r"html5?", "CSS": r"css3?|tailwind|flexbox", "CSV": r"csv",
}
_MAKE = re.compile(r"\b(write|create|make|generate|build|design|give|show|style|layout)\b", _I)
_HANDLE = re.compile(r"\b(read|reads|reading|parse|parsing|load|loading|save|saving|dump|convert|converting|"
                     r"serialize|deserialize|scrape|scraping|extract|validate|open|import|export|send|fetch|request)\b", _I)

# 5 — any other word in an unmistakable phrase
_GENERIC = [
    re.compile(r"\bhello,? world (?:program )?in ([\w#+.-]+)", _I),
    re.compile(r"\b(?:written|coded|implemented|programmed) in ([\w#+.-]+)", _I),
    re.compile(r"\blanguage (?:called|named) ([\w#+.-]+)", _I),
    re.compile(r"\b([\w#+.-]+) (?:programming )?language\b", _I),
]
_NOT_LANGUAGES = set("""
a an the this that these those my your our their his her its any some every each other another same different
plain simple natural spoken human sign body foreign native official first second new old modern favorite
favourite programming coding computer scripting query markup functional object oriented procedural low high level
compiled interpreted typed dynamic static general purpose domain specific english spanish french german chinese
japanese korean hindi bengali bangla arabic russian portuguese italian turkish words sentences one two three line
lines place order detail full short long good bad which what whatever whichever such own style form way terms
python py pythonic love strong clear formal
""".split())

PYTHON = re.compile(
    r"\b(?:python\d?|pandas|numpy|scipy|matplotlib|seaborn|sklearn|scikit-learn|tensorflow|keras|pytorch|torch|"
    r"opencv|cv2|tkinter|pygame|asyncio|sqlalchemy|beautifulsoup|bs4|selenium|scrapy|pydantic|django|flask|"
    r"fastapi|pytest|jupyter|pip|conda|venv)\b|\.py\b", _I)
_TARGET = re.compile(r"\b(to|into)\s+(?:the\s+)?$", _I)

_ALT = r"(?<![\w#+]){}(?![\w#+])"
_PATTERNS = (
    [(name, re.compile(_ALT.format(f"(?:{alts})"), _I)) for name, alts in KNOWN.items()]
    + [(name, re.compile(_ALT.format(f"(?:{alts})"), _I)) for name, alts in FRAMEWORKS.items()]
    + [(name, re.compile(rf"\b{_BEFORE} (?P<w>{alts})(?![\w#+])", _I)) for name, alts in SHORT.items()]
    + [(name, re.compile(rf"(?<![\w#+])(?P<w>{alts}) {_AFTER}\b", _I)) for name, alts in SHORT.items()]
    + [(name, re.compile(rf"(?<![\w#+])(?:{alts}) {_LANGUAGE_WORD}\b", _I)) for name, alts in WORDY.items()]
)
_FORMAT_PATTERNS = [(name, re.compile(_ALT.format(f"(?:{alts})"), _I)) for name, alts in FORMATS.items()]


def _tag(name: str) -> str:
    """Code-block tag for a language name: C++ → cpp, C# → csharp, Objective-C → objective-c."""
    special = {"C++": "cpp", "C#": "csharp", "F#": "fsharp", "Assembly": "asm", "Wolfram Language": "wolfram"}
    return special.get(name, re.sub(r"[^\w+#-]", "", name.lower().replace(" ", "-")))


def _found(message: str) -> list:
    """(position, name) of every language named in the message."""
    found = []
    # (position, -length, name): sorted, the earliest wins and at the same spot the
    # longest ("in c sharp" → C#, not C)
    for name, pattern in _PATTERNS:
        match = pattern.search(message)
        if match:   # the language word itself ("convert ... to go" → at "go")
            group = "w" if "w" in pattern.groupindex else 0
            found.append((match.start(group), -len(match.group(group)), name))
    if _MAKE.search(message) and not _HANDLE.search(message):
        for name, pattern in _FORMAT_PATTERNS:
            match = pattern.search(message)
            if match:
                found.append((match.start(), -len(match.group(0)), name))
    for pattern in _GENERIC:
        for match in pattern.finditer(message):
            word = match.group(1).strip(".-")
            if len(word) > 1 and word.lower() not in _NOT_LANGUAGES and not word.isdigit():
                found.append((match.start(1), -len(word), word.capitalize() if word.islower() else word))
    return [(start, name) for start, _, name in sorted(found)]


def detect_language(message: str) -> Optional[dict]:
    """{"name": "Haskell", "tag": "haskell"} for another language, None for Python / none named."""
    found = _found(message)
    if not found:
        return None
    for start, name in found:
        if _TARGET.search(message[:start]):
            return {"name": name, "tag": _tag(name)}
    if PYTHON.search(message):
        return None
    name = found[0][1]
    return {"name": name, "tag": _tag(name)}


# ─── The open file's language (chat panel file reading, Phase 10.2) ──────────

# VS Code languageId → display name. Python ("python") and plain text aren't
# listed: they keep V's default. FILE_FORMATS count only when the file's text
# goes into the prompt ("fix this" in config.yaml → YAML; "write a function
# that ..." with README.md open stays Python). Ids not listed are treated as
# formats too ("pip-requirements" must not turn a request into its language).
FILE_LANGUAGES = {
    "javascript": "JavaScript", "javascriptreact": "JavaScript", "typescript": "TypeScript",
    "typescriptreact": "TypeScript", "c": "C", "cpp": "C++", "cuda-cpp": "CUDA", "cuda": "CUDA", "arduino": "C++",
    "csharp": "C#", "java": "Java", "go": "Go", "rust": "Rust", "php": "PHP", "ruby": "Ruby", "shellscript": "Bash",
    "powershell": "PowerShell", "bat": "Batch", "sql": "SQL", "lua": "Lua", "perl": "Perl", "perl6": "Raku",
    "raku": "Raku", "r": "R", "swift": "Swift", "kotlin": "Kotlin", "dart": "Dart", "haskell": "Haskell",
    "elixir": "Elixir", "erlang": "Erlang", "fsharp": "F#", "clojure": "Clojure", "scala": "Scala",
    "groovy": "Groovy", "vb": "Visual Basic", "objective-c": "Objective-C", "objective-cpp": "Objective-C++",
    "julia": "Julia", "matlab": "MATLAB", "ocaml": "OCaml", "coffeescript": "CoffeeScript", "zig": "Zig",
    "nim": "Nim", "solidity": "Solidity", "vyper": "Vyper", "verilog": "Verilog", "systemverilog": "SystemVerilog",
    "vhdl": "VHDL", "glsl": "GLSL", "hlsl": "HLSL", "wgsl": "WGSL", "opencl": "OpenCL", "terraform": "Terraform",
    "hcl": "Terraform", "dockerfile": "Dockerfile", "makefile": "Makefile", "cmake": "CMake", "vue": "Vue",
    "svelte": "Svelte", "asm": "Assembly", "nasm": "Assembly", "masm": "Assembly", "gas": "Assembly",
    "arm": "Assembly", "arm64": "Assembly", "riscv": "Assembly", "mips": "Assembly",
    "asm-intel-x86-generic": "Assembly", "asm-collection": "Assembly", "fortran": "Fortran",
    "fortran-modern": "Fortran", "fortranfreeform": "Fortran", "fortranfixedform": "Fortran", "cobol": "COBOL",
    "pascal": "Pascal", "objectpascal": "Delphi", "ada": "Ada", "prolog": "Prolog", "lisp": "Lisp",
    "commonlisp": "Lisp", "scheme": "Scheme", "racket": "Racket", "elm": "Elm", "purescript": "PureScript",
    "reason": "Reason", "rescript": "ReScript", "crystal": "Crystal", "d": "D", "v": "Vlang", "gdscript": "GDScript",
    "graphql": "GraphQL", "proto": "Protocol Buffers", "proto3": "Protocol Buffers", "nix": "Nix", "apex": "Apex",
    "abap": "ABAP", "sas": "SAS", "stata": "Stata", "wolfram": "Wolfram Language", "tcl": "Tcl", "awk": "AWK",
    "ahk": "AutoHotkey", "autohotkey": "AutoHotkey", "applescript": "AppleScript", "haxe": "Haxe", "mojo": "Mojo",
    "gleam": "Gleam", "odin": "Odin", "cython": "Cython", "move": "Move", "cairo": "Cairo", "lean": "Lean",
    "lean4": "Lean", "coq": "Coq", "idris": "Idris", "agda": "Agda", "smalltalk": "Smalltalk", "apl": "APL",
    "forth": "Forth", "pony": "Pony", "chapel": "Chapel", "hare": "Hare", "carbon": "Carbon", "gdshader": "GLSL",
    "shaderlab": "ShaderLab", "powerquery": "Power Query M", "dax": "DAX", "kusto": "KQL", "bicep": "Bicep",
    "puppet": "Puppet", "razor": "Razor", "aspnetcorerazor": "Razor", "vbscript": "Visual Basic",
}
FILE_FORMATS = {
    "json": "JSON", "jsonc": "JSON", "jsonl": "JSON Lines", "json5": "JSON5", "yaml": "YAML",
    "dockercompose": "YAML", "ansible": "YAML", "helm": "YAML", "github-actions-workflow": "YAML", "xml": "XML",
    "xsl": "XSLT", "toml": "TOML", "markdown": "Markdown", "html": "HTML", "django-html": "HTML",
    "jinja": "Jinja", "handlebars": "Handlebars", "pug": "Pug", "css": "CSS", "scss": "SCSS", "sass": "Sass",
    "less": "Less", "latex": "LaTeX", "tex": "LaTeX", "bibtex": "BibTeX", "restructuredtext": "reStructuredText",
    "csv": "CSV", "tsv": "TSV", "ini": "INI", "properties": "Properties", "dotenv": "dotenv",
    "pip-requirements": "pip requirements",
}
_PLAIN_FILES = {"", "python", "plaintext", "log", "ignore", "diff", "git-commit", "git-rebase", "search-result",
                "code-text-binary", "snippets", "jupyter", "code-workspace"}


def file_language(language_id: str) -> Optional[dict]:
    """
    {"name", "tag", "format"} for the open file's VS Code languageId; None for
    Python and plain text (V's default). format True = counts only when the
    file's text goes into the prompt.
    """
    key = (language_id or "").strip().lower()
    if key in _PLAIN_FILES:
        return None
    if key in FILE_LANGUAGES:
        name = FILE_LANGUAGES[key]
        return {"name": name, "tag": _tag(name), "format": False}
    name = FILE_FORMATS.get(key) or key.replace("-", " ").title()
    return {"name": name, "tag": key if key in FILE_FORMATS else _tag(name), "format": True}


def names_python(message: str) -> bool:
    """The message names Python or one of its libraries ("do it in python", "with pandas")."""
    return bool(PYTHON.search(message))
