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
      inherit (pkgs) poetry2nix lib fetchurl;
      inherit (pkgs.cudaPackages) cudatoolkit;
      inherit (pkgs.linuxPackages) nvidia_x11;
      python = pkgs.python39;
      pythonEnv = poetry2nix.mkPoetryEnv {
        inherit python;
        projectDir = ./.;
        preferWheels = true;
        overrides = poetry2nix.overrides.withDefaults (pyfinal: pyprev: rec {
          # Use cuda-enabled pytorch as required
          torch =
            if useCuda
            then
              # Override the nixpkgs bin version instead of
              # poetry2nix version so that rpath is set correctly.
              pyprev.pytorch-bin.overridePythonAttrs
              (old: {
                inherit (old) pname version;
                src = fetchurl {
                  url = "https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp39-cp39-linux_x86_64.whl";
                  sha256 = "sha256-20V6gi1zYBO2/+UJBTABvJGL3Xj+aJZ7YF9TmEqa+sU=";
                };
              })
            else pyprev.torch;

          # Provide non-python dependencies.
        });
      };
    in {
      devShell = pkgs.mkShell {
        buildInputs =
          [pythonEnv]
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
