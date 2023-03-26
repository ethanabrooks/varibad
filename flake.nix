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
        overrides = poetry2nix.overrides.withDefaults (pyfinal: pyprev: let
          inherit (pyprev) buildPythonPackage fetchPypi;
        in rec {
          # Use cuda-enabled pytorch as required
          setuptools-scm = buildPythonPackage rec {
            pname = "setuptools-scm";
            version = "7.0.5";

            src = fetchPypi {
              pname = "setuptools_scm";
              inherit version;
              sha256 = "sha256-Ax4Tr3cdb4krlBrbbqBFRbv5Hrxc5ox4qvP/9uH7SEQ=";
            };

            propagatedBuildInputs = with pyfinal; [
              packaging
              #typing-extensions
              tomli
              setuptools
            ];

            pythonImportsCheck = [
              "setuptools_scm"
            ];

            # check in passthru.tests.pytest to escape infinite recursion on pytest
            doCheck = false;

            passthru.tests = {
              pytest = pkgs.callPackage ./tests.nix {};
            };

            #meta = with lib; {
            #homepage = "https://github.com/pypa/setuptools_scm/";
            #description = "Handles managing your python package versions in scm metadata";
            #license = licenses.mit;
            #maintainers = with maintainers; [SuperSandro2000];
            #};
          };
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
