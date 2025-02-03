# PySFNOCI
A library of Spin-Flip Non-orthogonal Configuration Interaction (SF-NOCI) with and without grouped bath.

Installation
------------

* Prerequisites
    - [PySCF](https://github.com/pyscf/pyscf): v2.8.0

* Compile C part of PySFNOCI code
    ```
    cd /path/to/PySFNOCI/pyscf/sfnoci
    gcc -o SFNOCI_contract.so -shared -fPIC SFNOCI_contract.c
    ```

* Add PySFNOCI top-level directory to your `PYSCF_EXT_PATH`
    ```
    export PYSCF_EXT_PATH="/path/to/PySFNOCI:$PYSCF_EXT_PATH"
    ```
