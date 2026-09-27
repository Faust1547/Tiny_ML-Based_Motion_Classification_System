# C & C++

## Overview
本專案結合 Arduino DUE 與 MPU6050 建立嵌入式動作辨識系統，透過 GPIO 實作 I²C 通訊，擷取三軸加速度與角速度資料，經前處理與 INT8 量化後，使用預先訓練的 1D-CNN 模型進行動作分類，並針對記憶體使用量與推論延遲進行優化。

 ## Execution
依照接線圖完成 Arduino DUE、MPU6050 與按鈕的硬體連接。將所有程式檔案放在同一個 Arduino 專案資料夾中，編譯並上傳 ESD.ino。

<img width="350" height="400" alt="image" src="https://github.com/user-attachments/assets/0dfbfeaa-11c8-4a09-9aae-9a0fa7ce1c1c" />

上傳完成後，開啟 Serial Monitor，將 Baud Rate 設為 115200。系統完成初始化後，按下按鈕即可開始擷取動作資料，並在推論結束後輸出分類結果。

 ## Process Flow
系統依序執行 MPU6050 初始化、動作資料擷取、定點數前處理及 1D-CNN 推論，並透過靜止偵測與預測信心門檻，決定最終分類結果。
 
<img width="2982" height="1539" alt="image" src="https://github.com/user-attachments/assets/73426f6e-2795-4885-b4b3-4266d694adb7" />

 ## File Description
|File|Description|
|---|---|
|ESD.ino|主程式，整合 MPU6050 初始化、資料擷取、前處理、神經網路推論及結果輸出|
|I2C_GPIO.cpp|GPIO 模擬 I²C 通訊之函式實作|
|I2C_GPIO.h|GPIO 模擬 I²C 通訊之函式宣告|
|nn_ops.cpp|神經網路運算函式，包含卷積、池化及全連接層等運算|
|nn_weights.cpp|預先訓練並量化後的神經網路權重及相關參數|
