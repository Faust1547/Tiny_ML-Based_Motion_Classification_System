# Five-Stage Pipelined Processor

## Overview
本專案延伸課堂實作之五級 Pipeline CPU，將原先的 IM 與 DM 暫存器替換為實際 256 × 32 SRAM，並於後續合成進一步導入 Design for Testability (DFT) 與 Scan Chain，使內部暫存器具備較佳的 Controllability 與 Observability，並透過 ATPG 進行 Stuck-at Fault 測試與 Fault Coverage 評估。

## Architecture
五階 Pipelined CPU 架構，整合 SRAM、Hazard Detection、Forwarding 與 Scan-based DFT。
 
<img width="4113" height="1188" alt="image" src="https://github.com/user-attachments/assets/ada8b590-36e3-47a3-8c45-23de0a7b966b" />



## Results
1. Post-sim 執行結果。
2. Automatic test pattern generation 執行結果。
3. TSMC 90nm 1P9M 實體設計之時序、面積、功耗紀錄，以及晶片實現結果與 Partition 表示。
