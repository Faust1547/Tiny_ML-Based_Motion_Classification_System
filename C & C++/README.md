# C & C++

## Overview
本專案結合 Arduino 與 MPU 6050 設計動作辨識系統，透過神經網路學習前後、上下、旋轉動作的路徑變化從而進行辨識，並透過 GPIO 讓 Arduino 與 MPU 進行資料傳輸。

 ## Execution
將硬體如接線圖完成連接：

<img width="350" height="400" alt="image" src="https://github.com/user-attachments/assets/0dfbfeaa-11c8-4a09-9aae-9a0fa7ce1c1c" />

而後將所有程式放在同層資料夾中並啟動主程式 `ESD.ino`，而後連接 Arduino DUE 並燒入，之後按下按鈕後即開始錄製動作並進行判定。

 ## Process Flow
<img width="2982" height="1539" alt="image" src="https://github.com/user-attachments/assets/73426f6e-2795-4885-b4b3-4266d694adb7" />

 ## File Description
|File|Description|
|---|---|
|ESD.ino|主程式，包含 GPIO 設定、MPU 6050 初始化、資料收集與處理|
|I2C_GPIO.cpp|GPIO 函式庫設定檔案|
|I2C_GPIO.h|GPIO 函式庫標頭檔案|
|nn_ops.cpp|神經網路運算程式|
|nn_weights.cpp|神經網路權重資料|
