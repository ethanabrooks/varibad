{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/a983cc62cc2345443620597697015c1c9c4e5b06";
    utils.url = "github:numtide/flake-utils/93a2b84fc4b70d9e089d029deacc3583435c2ed6";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
  }: let
    out = system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };
      inherit (pkgs) poetry2nix;
      overrides = pyfinal: pyprev: rec {
        mujoco-py =
          (pyprev.mujoco-py.override {
            preferWheel = false;
            format = "pyproject";
          })
          .overridePythonAttrs (old: {
            patches = (old.patches or []) ++ [./mujoco-py.patch];
            env.NIX_CFLAGS_COMPILE = "-I${pkgs.mujoco}/include/";
            preBuild = ''
              # this line removes a bug where value of $HOME is set to a non-writable /homeless-shelter dir
              echo 'MUJOCO PATH:' ${pkgs.mujoco}
              export MUJOCO_PY_MUJOCO_PATH="${pkgs.mujoco}"
              #export MUJOCO_PY_SO_PATH=/tmp
              export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.mujoco}/bin:${pkgs.mujoco}/include
            '';
            #postBuild = "echo 'OUT'; echo $out; cd $out; echo 'REALPATH'; realpath .; echo 'LS'; ls; exit 1;";
            buildInputs =
              (old.buildInputs or [])
              ++ (with pyprev; [
                setuptools
                fasteners
                numpy
                cython
                cffi
                pkgs.glew
              ]);
            propagatedBuildInputs = (old.propagatedBuildInputs or []) ++ [pkgs.mujoco];
          });
        torch = pyprev.pytorch-bin.overridePythonAttrs (old: {
          src = pkgs.fetchurl {
            url = "https://download.pytorch.org/whl/cu115/torch-1.11.0%2Bcu115-cp39-cp39-linux_x86_64.whl";
            sha256 = "sha256-64HQZ7vP6ETJXF0n4myXqWqJNCfMRosiWerw7ZPaHH0=";
          };
        });
      };
      poetryEnv = pkgs.poetry2nix.mkPoetryEnv {
        python = pkgs.python39;
        projectDir = ./.;
        preferWheels = true;
        overrides = poetry2nix.overrides.withDefaults overrides;
      };
    in {
      devShell = pkgs.mkShell {
        LD_LIBRARY_PATH = "${pkgs.linuxPackages.nvidia_x11}/lib";
        buildInputs = with pkgs; [
          alejandra
          poetry
          poetryEnv
        ];
        PYTHONBREAKPOINT = "ipdb.set_trace";
        shellHook = ''
          set -o allexport
          source .env
          set +o allexport
        '';
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
