## Inference Results
|Left & Right|Circle|Up & Down|
|---|---|---|
|<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/6d1e497d-2f31-4e0a-bc18-afedc0563d7e" />|<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/e273d9f8-2857-440e-88df-d0d68efd1820" />|<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/cc39ffbe-a095-4914-aebe-f6cc4720b075" />|


## Optimization Results
相較於課堂實作的基準版本，本專案改良版本引入了多項改良與額外設計。

| Optimization Item | Improvement | Trade-off / Overhead |
|---|---|---|
| RTOS-based task separation | 改善任務排程和系統的可擴展性。 | 與 Busy-wait 排程相比 Sampling jitter 會稍微變差。 |
| SRAM reduction | SRAM 用量減少 34.28 % | 無 |
| Inference latency reduction | 推論延遲降低 27.09 % | 無 |
| Decision logic enhancement | 新增了 `Static` 與 `Uncertain` 的辨識結果，並且在 `Static` 時能夠跳過 NN 流程與系統待機| 額外的 log 可能會增加串列輸出的延遲時間 |
