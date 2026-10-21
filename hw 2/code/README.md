# Homework 2: Decision stump（Q12–Q16）

使用共用 Python 函式庫及每題獨立實作，無需 Notebook 執行狀態。程式碼、註解與 Google style docstring 均使用英文。Q11 為理論題，相關公式說明如下，沒有另外的實驗入口。

## 環境與重現方式

使用 Python 3.12 與 NumPy 1.26.4。在本資料夾開啟 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_all.py --output results.json
```

若已安裝指定版本的 NumPy，可直接執行：

```powershell
python run_all.py --output results.json
```

每題產生 10,000 組獨立訓練資料。預設 base seed 為 2026，各題實際 seed 為 2026 + 題號。亂數使用 NumPy PCG64；每組資料依序抽取 x 和翻轉標籤的亂數，包含 tau = 0 的情況。固定套件版本、種子、抽樣順序可重現相同結果。無需下載資料。

## 檔案

| 檔案 | 用途 |
| --- | --- |
| decision_stump.py | 共用資料生成、訓練、解析 E_out，以及 CLI 參數與 JSON 輸出工具 |
| q12.py | N = 3，tau = 0 |
| q13.py | N = 30，tau = 0 |
| q14.py | N = 3，tau = 0.1 |
| q15.py | N = 30，tau = 0.1 |
| q16.py | N = 300，tau = 0.1 |
| run_all.py | 呼叫各題的 solve() 並輸出完整結果 |
| results.json | 預設設定的實際執行結果 |
| test_decision_stump.py | 暴力搜尋比對、邊界、公式與可重現性檢查 |
| requirements.txt | 固定 NumPy 版本 |

各題的 `solve()` 自行設定 N、tau、亂數種子與選項，實作 10,000 次資料生成與訓練迴圈，計算 median(E_out - E_in) 並選出答案。`decision_stump.py` 僅提供可重用函式，不再包含題目設定表或題號分派。`run_all.py` 直接呼叫各題實作，`solve()` 不會自行輸出或寫入檔案。

各題可單獨執行，例如：

```powershell
python q12.py
python q16.py --output q16_results.json
python q14.py --seed 2026 --trials 10000
```

個別執行與整批執行使用相同的各題種子，因此結果一致。`--output` 的相對路徑以目前工作目錄為準；未指定時只輸出至終端機。提交某題原始碼時，需同時提供該題入口與共用的 decision_stump.py。

## 演算法與公式

1. x 均勻分布於 [-1, 1]，y = sign(x)，以 tau 的機率獨立翻轉 y。sign(0) = -1。
2. 將訓練資料排序；候選門檻僅包含 -1 及相鄰不同 x 的中點，方向為 -1 與 +1。
3. 使用正標籤的 prefix counts 計算各門檻錯誤數。s = +1 時，錯誤數為「左側正標籤數 + 右側負標籤數」；s = -1 時為 N 減去前者。
4. 先選最少錯誤數，再從平手候選選出最小的 s + theta。排序為 O(N log N)，排序後評估為 O(N)。
5. 每次計算 E_out(g, tau) - E_in(g)，最後取 10,000 個差值的中位數，再選最接近的選項。偶數樣本中位數取中央兩值平均。

對 theta 位於 [-1, 1]，s = +1 時，與 sign(x) 不同的區間長度為 |theta|，均勻分布總長度為 2，所以乾淨標籤的錯誤率為 |theta| / 2。s = -1 則為 1 - |theta| / 2。加入獨立標籤雜訊後：

```text
clean_error = |theta| / 2             if s = +1
clean_error = 1 - |theta| / 2         if s = -1
E_out = tau + (1 - 2 * tau) * clean_error
```

因此 Q11.1 答案是選項 2：|theta| / 2。程式使用解析 E_out，不需要額外測試資料，也不會增加測試抽樣誤差。

## 驗證

```powershell
python -m unittest -v test_decision_stump.py
```

驗證會將最佳化訓練結果與直接枚舉、逐筆預測的版本比對，包含重複座標、x = -1、只有一筆資料與平手情況；另使用密集均勻網格確認 E_out 公式。

實際中位數與選項請見 results.json；不同亂數種子的實驗中位數可能略有差異。


## 本次重現結果

已使用 Python 3.12.7、NumPy 1.26.4 驗證，base seed = 2026，每題 10,000 次。四項 unittest 均通過。

| 題目 | N | tau | 實際 seed | median(E_out - E_in) | 最接近選項 | 答案 |
| --- | --- | --- | --- | --- | --- | --- |
| Q12 | 3 | 0 | 2038 | 0.1047779532 | 0.10 | 1 |
| Q13 | 30 | 0 | 2039 | 0.0114434750 | 0.01 | 2 |
| Q14 | 3 | 0.1 | 2040 | 0.2091477254 | 0.20 | 3 |
| Q15 | 30 | 0.1 | 2041 | 0.0374596427 | 0.04 | 5 |
| Q16 | 300 | 0.1 | 2042 | 0.0038537073 | 0.004 | 3 |

完整精度、參數與選項均保存在 results.json。


## WSL rerun (October 20, 2026)

All tests, the batch program, and Q12-Q16 entry points were rerun successfully using Python 3.13.5 and NumPy 2.2.6. The numerical results match the earlier Windows run. The updated results.json records NumPy 2.2.6; requirements.txt retains the original Python 3.12 / NumPy 1.26.4 setup.

```bash
cd "/home/daniel/Study/ML-2026/hw 2/code"
/home/daniel/miniconda3/bin/python -B -m unittest -v test_decision_stump.py
/home/daniel/miniconda3/bin/python -B run_all.py --output results.json
```

See [the English teaching report](../report.md) for step-by-step derivations, implementation explanations, answer analysis, and WSL reproduction details. Execution evidence is in results/wsl_run.json and results/execution_log.txt.


## Current environment: ML2026

The latest verification used the WSL conda environment ML2026 with Python 3.15.0 and NumPy 2.5.4. All four unit tests passed and Q12-Q16 each completed 10,000 trials. Every experiment record exactly matched the previous run. requirements.txt now pins numpy==2.5.4; results.json contains this latest run. Earlier environment descriptions above document historical runs.

```bash
conda activate ML2026
cd "/home/daniel/Study/ML-2026/hw 2/code"
python -B -m unittest -v test_decision_stump.py
python -B run_all.py --output results.json
```

Verification evidence: results/ml2026_run.json, results/ml2026_execution_log.txt, and results/ml2026_results.json.
