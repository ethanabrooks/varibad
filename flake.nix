{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/6141b8932a5cf376fe18fcd368cecd9ad946cb68";
    utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    utils,
  }: let
    out = system: let
      useCuda = system == "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.cudaSupport = useCuda;
        config.allowUnfree = true;
      };
      inherit (pkgs) poetry2nix lib stdenv fetchurl fetchFromGitHub fetchPypi;
      inherit (pkgs.cudaPackages) cudatoolkit;
      inherit (pkgs.linuxPackages) nvidia_x11;
      python = pkgs.python39;
      pythonEnv = poetry2nix.mkPoetryEnv {
        inherit python;
        projectDir = ./.;
        preferWheels = true;
        overrides = poetry2nix.overrides.withDefaults (pyfinal: pyprev: rec {
          # Use cuda-enabled pytorch as required
          matplotlib = pyprev.matplotlib.overridePythonAttrs (old: let
            enableTk = true;
          in {
            inherit (old) version pname;
            #version = "3.5.3";
            #pname = "matplotlib";
            format = "setuptools";

            src = fetchPypi {
              inherit (old) version pname;
              sha256 = "sha256-M5ysSLgN28i/0F2q4KOnNBRlGoWWkEwqiBz9Httl8mw=";
            };

            XDG_RUNTIME_DIR = "/tmp";

            nativeBuildInputs = with pkgs; [
              pkg-config
              setuptools-scm
              setuptools-scm-git-archive
            ];

            buildInputs = with pkgs;
              [
                which
                sphinx
              ]
              ++ lib.optionals enableGhostscript [
                ghostscript
              ]
              ++ lib.optionals stdenv.isDarwin [
                Cocoa
              ];

            propagatedBuildInputs = with pkgs;
              [
                certifi
                cycler
                fonttools
                freetype
                kiwisolver
                libpng
                mock
                numpy
                packaging
                pillow
                pyparsing
                python-dateutil
                pytz
                qhull
                tornado
              ]
              ++ lib.optionals enableGtk3 [
                cairo
                gobject-introspection
                gtk3
                pycairo
                pygobject3
              ]
              ++ lib.optionals enableTk (with pkgs; [
                xorg.libX11
                tcl
                tk
                tkinter
              ])
              ++ lib.optionals enableQt [
                pyqt5
              ];

            passthru.config = {
              directories = {basedirlist = ".";};
              libs =
                {
                  system_freetype = true;
                  system_qhull = true;
                }
                // lib.optionalAttrs stdenv.isDarwin {
                  # LTO not working in darwin stdenv, see #19312
                  enable_lto = false;
                };
            };

            MPLSETUPCFG = with pkgs; writeText "mplsetup.cfg" (lib.generators.toINI {} passthru.config);

            # Matplotlib tries to find Tcl/Tk by opening a Tk window and asking the
            # corresponding interpreter object for its library paths. This fails if
            # `$DISPLAY` is not set. The fallback option assumes that Tcl/Tk are both
            # installed under the same path which is not true in Nix.
            # With the following patch we just hard-code these paths into the install
            # script.
            postPatch = let
              tcl_tk_cache = with pkgs; ''"${tk}/lib", "${tcl}/lib", "${lib.strings.substring 0 3 tk.version}"'';
            in
              lib.optionalString enableTk ''
                sed -i '/self.tcl_tk_cache = None/s|None|${tcl_tk_cache}|' setupext.py
              ''
              + lib.optionalString (stdenv.isLinux) ''
                # fix paths to libraries in dlopen calls (headless detection)
                substituteInPlace src/_c_internal_utils.c \
                  --replace libX11.so.6 ${pkgs.xorg.libX11}/lib/libX11.so.6 \
                  --replace libwayland-client.so.0 ${pkgs.wayland}/lib/libwayland-client.so.0
              ''
              +
              # avoid matplotlib trying to download dependencies
              ''
                echo "[libs]
                system_freetype=true
                system_qhull=true" > mplsetup.cfg
                substituteInPlace setup.py \
                  --replace "setuptools_scm>=4,<7" "setuptools_scm>=4"
              '';

            # Matplotlib needs to be built against a specific version of freetype in
            # order for all of the tests to pass.
            doCheck = false;
          });
          torch =
            if useCuda
            then
              # Override the nixpkgs bin version instead of
              # poetry2nix version so that rpath is set correctly.
              pyprev.pytorch-bin.overridePythonAttrs
              (old: {
                inherit (old) pname version;
                src = fetchurl {
                  url = "https://download.pytorch.org/whl/cu115/torch-1.11.0%2Bcu115-cp39-cp39-linux_x86_64.whl";
                  sha256 = "sha256-64HQZ7vP6ETJXF0n4myXqWqJNCfMRosiWerw7ZPaHH0=";
                };
              })
            else pyprev.torch;

          # Provide non-python dependencies.
        });
      };
    in {
      devShell = pkgs.mkShell {
        buildInputs =
          [pythonEnv pkgs.python39Packages.poetry]
          ++ lib.optionals useCuda [
            nvidia_x11
            cudatoolkit
          ];
        shellHook =
          ''
            export pythonfaulthandler=1
            export pythonbreakpoint=ipdb.set_trace
            set -o allexport
            source .env
            set +o allexport
          ''
          + pkgs.lib.optionalString useCuda ''
            export CUDA_PATH=${cudatoolkit.lib}
            export LD_LIBRARY_PATH=${cudatoolkit.lib}/lib:${nvidia_x11}/lib
            export EXTRA_LDFLAGS="-l/lib -l${nvidia_x11}/lib"
            export EXTRA_CCFLAGS="-i/usr/include"
          '';
      };
    };
  in
    with utils.lib; eachSystem defaultSystems out;
}
