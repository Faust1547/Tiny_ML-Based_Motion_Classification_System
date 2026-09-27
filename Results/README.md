## Inference Results
|Left & Right|Circle|Up & Down|
|---|---|---|
|<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/6d1e497d-2f31-4e0a-bc18-afedc0563d7e" />|<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/e273d9f8-2857-440e-88df-d0d68efd1820" />|<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/cc39ffbe-a095-4914-aebe-f6cc4720b075" />|


## Optimization Results
相較於課堂實作的基準版本，本專案改良版本引入了多項改良與額外設計。

| Optimization Item | Improvement | Trade-off / Overhead |
|---|---|---|
| RTOS-based task separation | 改善任務排程和系統的可擴展性 | 與 Busy-wait 排程相比 Sampling jitter 會稍微變差 |
| SRAM reduction | SRAM 用量減少 34.28 % | — |
| Inference latency reduction | 推論延遲降低 27.09 % | — |
| Decision logic enhancement | 新增靜止偵測與低信心分類機制；偵測到靜止狀態時跳過 NN 推論，降低不必要運算，並利用 Prediction Margin 判斷是否輸出 Uncertain| 額外的 log 可能會增加串列輸出的延遲時間 |
