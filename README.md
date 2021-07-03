# Fuzzing for adversarial examples

1. Make sure you have installed libFuzzer (clang-11) and optionally the LT-attack (executable should be in the root of this repo as `lt_attack`), the Gurobi optimizer, AFL++ and honggfuzz
2. Set the appropriate paths in `python/constants.py`
3. Download the data.
   - breast-cancer, diabetes, ijcnn1, covertype from LibSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/.
   - higgs 90.000 instances from OPENML: https://www.openml.org/search?type=data
   - (F)MNIST: original datasets
   - Make sure the data is put in the right location (inspect `python/load_data.py`)
   - Other datasets can be added easily. Just specify them in `python/datasets.py` and `python/load_data.py`
4. Adjust the settings in `python/constants.py` according to your wishes
5. There are 2 command line tools: `python/run_fate.py` and `python/automate.py`. Execute with `-h` flag for detailed descriptions and options.
6. Run `python3 python/run_fate.py [dataset_name] [model_type] --reload` once for each dataset
7. Using `--reload` is only necessary for the first run for each dataset. It will generate/train the model to be fuzzed. Afterwards, do not provide `--reload` for speedup (training the model is not necessary anymore) and consistency.
8. Optionally, the baseline methods can be run via `python/run_milp.py` and `python/run_zhang.py`
9. Most settings in `python/constants.py` (everything that does not have to do with the training of the model) can be adjusted without having to specify `--reload` again.
