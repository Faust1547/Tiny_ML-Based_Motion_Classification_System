#include "I2C_GPIO.h"
#include "nn_ops.h"
#include <math.h>

#include <FreeRTOS.h>
#include <task.h>
#include <queue.h>

#ifndef pdMS_TO_TICKS
#define pdMS_TO_TICKS(ms) ((TickType_t)((ms) / portTICK_PERIOD_MS))
#endif

#define I2C_use_GPIO 1

static const uint8_t MPU_ADDR = 0x68;

#define PWR_MGMT_1    0x6B
#define ACCEL_XOUT_H  0x3B

#define LED_PIN       6
#define BTN_PIN       12

#define DEBOUNCE_MS   30

#define SAMPLE_RATE_HZ      100
#define SAMPLE_INTERVAL_US  10000UL   // 100 Hz = 10 ms
#define RECORD_SECONDS      1.5
#define NUM_SAMPLES         150       // 100 Hz * 1.5 sec

// Debug switches
#define ENABLE_SAMPLE_LOGGING 0
#define ENABLE_JITTER_REPORT  1
#define ENABLE_INPUT_SMOOTHING 1

// Static / uncertain decision thresholds.
#define STATIC_ACC_DELTA_TH 1000UL
#define STATIC_GYRO_ABS_TH  1000UL
#define LOW_CONF_MARGIN_TH  20

bool isRecording = false;

TaskHandle_t captureTaskHandle = NULL;
TaskHandle_t buttonTaskHandle = NULL;
TaskHandle_t inferenceTaskHandle = NULL;
TaskHandle_t loggingTaskHandle = NULL;
QueueHandle_t logQueueHandle = NULL;

#define BUTTON_TASK_PRIORITY     (tskIDLE_PRIORITY + 1)
#define LOGGING_TASK_PRIORITY    (tskIDLE_PRIORITY + 1)
#define INFERENCE_TASK_PRIORITY  (tskIDLE_PRIORITY + 2)
#define CAPTURE_TASK_PRIORITY    (tskIDLE_PRIORITY + 3)
#define BUTTON_TASK_STACK_WORDS     256
#define LOGGING_TASK_STACK_WORDS    640
#define INFERENCE_TASK_STACK_WORDS  2048
#define CAPTURE_TASK_STACK_WORDS    2048
#define LOG_QUEUE_LENGTH            32

unsigned long lastDebounceMs = 0;
int lastButtonReading = HIGH;
int buttonState = HIGH;

uint32_t sampleId = 0;
unsigned long totalStartUs = 0;

long jitterMaxAbsUs = 0;
long jitterSumAbsUs = 0;
unsigned long prevSampleUs = 0;
int64_t intervalErrorSumUs = 0;
int64_t intervalErrorSqSumUs = 0;
uint32_t intervalCount = 0;

// Motion-gate statistics for static detection.
uint32_t accDeltaSum = 0;
uint32_t gyroAbsSum = 0;
uint32_t avgAccDelta = 0;
uint32_t avgGyroAbs = 0;
bool isStaticWindow = false;
uint32_t staticWindowCount = 0;
uint32_t uncertainWindowCount = 0;

// SRAM-optimized buffers:
// - raw_samples[][] is removed: raw int16 samples are quantized immediately.
// - pool1_out[][] is removed: conv1_pool_buf is reused after in-place maxpool.
int8_t nn_input[NN_INPUT_LEN][NN_INPUT_CH];
int8_t conv1_pool_buf[NN_INPUT_LEN][NN_CONV1_OUT_CH];
int8_t conv2_out[NN_INPUT_LEN / 2][NN_CONV2_OUT_CH];
int8_t gap_out[NN_CONV2_OUT_CH];
int8_t dense1_out[NN_DENSE1_OUT];
int8_t logits[NN_NUM_CLASSES];

enum LogMsgType {
  LOG_GET_READY,
  LOG_START,
  LOG_JITTER_REPORT,
  LOG_MOTION_REPORT,
  LOG_ERROR_STEP,
  LOG_INFERENCE_RESULT,
  LOG_SAMPLE_ROW,
  LOG_END
};

enum PredictionState {
  PRED_GESTURE = 0,
  PRED_STATIC = 1,
  PRED_UNCERTAIN = 2
};

struct LogMessage {
  LogMsgType type;
  uint32_t sample_id;
  int timestep;
  unsigned long t_us;
  int16_t ax, ay, az, gx, gy, gz;
  long jitter_max_abs_us;
  long jitter_avg_abs_us;
  double interval_jitter_std_us;
  uint32_t avg_acc_delta;
  uint32_t avg_gyro_abs;
  bool is_static;
  bool skip_nn;
  unsigned long inference_latency_us;
  unsigned long total_latency_us;
  int label;
  int prediction_state;
  int8_t logit0, logit1, logit2;
  int8_t margin;
  uint32_t static_windows;
  uint32_t uncertain_windows;
};

const char *LABEL_NAMES[NN_NUM_CLASSES] = {
  "circle",
  "left_right",
  "updown"
};

void MPU6050_wakeup();
bool readImu6_raw(int16_t *ax, int16_t *ay, int16_t *az,
                  int16_t *gx, int16_t *gy, int16_t *gz);
void buttonTask(void *pvParameters);
void captureTask(void *pvParameters);
void inferenceTask(void *pvParameters);
void loggingTask(void *pvParameters);
void handleButtonRTOS();
void recordOneSampleRTOS();
void runInferenceRTOS();
void enqueueLog(const LogMessage &msg);
uint32_t abs32(int32_t x);

uint32_t abs32(int32_t x) {
  return (x >= 0) ? (uint32_t)x : (uint32_t)(-x);
}

// Part1: I2C Communication and Data Collection
void MPU6050_wakeup() {
  I2C_start();

  if (!I2C_write_byte((MPU_ADDR << 1) | 0)) {
    I2C_stop();
    Serial.println("# ERROR: MPU6050 SLA+W NACK during wakeup");
    return;
  }

  if (!I2C_write_byte(PWR_MGMT_1)) {
    I2C_stop();
    Serial.println("# ERROR: PWR_MGMT_1 register NACK");
    return;
  }

  if (!I2C_write_byte(0x00)) {
    I2C_stop();
    Serial.println("# ERROR: wakeup data NACK");
    return;
  }

  I2C_stop();
  delay(10);
}

bool readImu6_raw(int16_t *ax, int16_t *ay, int16_t *az,
                  int16_t *gx, int16_t *gy, int16_t *gz) {
  uint8_t buf[14];

  I2C_start();

  if (!I2C_write_byte((MPU_ADDR << 1) | 0)) {
    I2C_stop();
    return false;
  }

  if (!I2C_write_byte(ACCEL_XOUT_H)) {
    I2C_stop();
    return false;
  }

  I2C_repeated_start();

  if (!I2C_write_byte((MPU_ADDR << 1) | 1)) {
    I2C_stop();
    return false;
  }

  for (int i = 0; i < 14; i++) {
    buf[i] = I2C_read_byte(i < 13);
  }

  I2C_stop();

  *ax = (int16_t)(((uint16_t)buf[0]  << 8) | buf[1]);
  *ay = (int16_t)(((uint16_t)buf[2]  << 8) | buf[3]);
  *az = (int16_t)(((uint16_t)buf[4]  << 8) | buf[5]);
  *gx = (int16_t)(((uint16_t)buf[8]  << 8) | buf[9]);
  *gy = (int16_t)(((uint16_t)buf[10] << 8) | buf[11]);
  *gz = (int16_t)(((uint16_t)buf[12] << 8) | buf[13]);

  return true;
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  pinMode(LED_PIN, OUTPUT);
  pinMode(BTN_PIN, INPUT_PULLUP);
  MPU6050_wakeup();
  analogWrite(LED_PIN, 0);

  Serial.println("Dataset collector ready. Final combined optimized version.");
  Serial.println("Push btn to start recording.");

  logQueueHandle = xQueueCreate(LOG_QUEUE_LENGTH, sizeof(LogMessage));
  if (logQueueHandle == NULL) {
    Serial.println("# ERROR: failed to create log queue");
    while (1) {}
  }

  xTaskCreate(buttonTask, "button", BUTTON_TASK_STACK_WORDS, NULL, BUTTON_TASK_PRIORITY, &buttonTaskHandle);
  xTaskCreate(captureTask, "capture", CAPTURE_TASK_STACK_WORDS, NULL, CAPTURE_TASK_PRIORITY, &captureTaskHandle);
  xTaskCreate(inferenceTask, "inference", INFERENCE_TASK_STACK_WORDS, NULL, INFERENCE_TASK_PRIORITY, &inferenceTaskHandle);
  xTaskCreate(loggingTask, "logging", LOGGING_TASK_STACK_WORDS, NULL, LOGGING_TASK_PRIORITY, &loggingTaskHandle);
  vTaskStartScheduler();

  Serial.println("# ERROR: FreeRTOS scheduler failed to start");
  while (1) {}
}

void loop() {
  // FreeRTOS 
}

void buttonTask(void *pvParameters) {
  (void)pvParameters;
  for (;;) {
    handleButtonRTOS();
    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

void captureTask(void *pvParameters) {
  (void)pvParameters;
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    if (!isRecording) {
      recordOneSampleRTOS();
    }
  }
}

void inferenceTask(void *pvParameters) {
  (void)pvParameters;
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    runInferenceRTOS();
  }
}

void loggingTask(void *pvParameters) {
  (void)pvParameters;
  LogMessage msg;
  for (;;) {
    if (xQueueReceive(logQueueHandle, &msg, portMAX_DELAY) == pdTRUE) {
      switch (msg.type) {
        case LOG_GET_READY:
          Serial.println("# GET_READY");
          break;

        case LOG_START:
          Serial.println("# START");
#if ENABLE_SAMPLE_LOGGING
          Serial.println("sample_id,timestep,t_us,ax,ay,az,gx,gy,gz");
#endif
          break;

        case LOG_JITTER_REPORT:
          Serial.print("# JITTER_MAX_ABS_US,");
          Serial.println(msg.jitter_max_abs_us);
          Serial.print("# JITTER_AVG_ABS_US,");
          Serial.println(msg.jitter_avg_abs_us);
          Serial.print("# INTERVAL_JITTER_STD_US,");
          Serial.println(msg.interval_jitter_std_us, 2);
          break;

        case LOG_MOTION_REPORT:
          //Serial.print("# AVG_ACC_DELTA,");
          //Serial.println(msg.avg_acc_delta);
          //Serial.print("# AVG_GYRO_ABS,");
          //Serial.println(msg.avg_gyro_abs);
          Serial.print("# MOTION_STATE,");
          Serial.println(msg.is_static ? "static" : "moving");
          break;

        case LOG_ERROR_STEP:
          Serial.print("# ERROR at timestep ");
          Serial.println(msg.timestep);
          break;

        case LOG_INFERENCE_RESULT:
          Serial.print("# INFERENCE_LATENCY_US,");
          Serial.println(msg.inference_latency_us);
          Serial.print("# TOTAL_LATENCY_US,");
          Serial.println(msg.total_latency_us);
          Serial.print("# SKIP_NN,");
          Serial.println(msg.skip_nn ? 1 : 0);

          if (!msg.skip_nn) {
            Serial.print("# LOGITS,");
            Serial.print((int)msg.logit0);
            Serial.print(",");
            Serial.print((int)msg.logit1);
            Serial.print(",");
            Serial.println((int)msg.logit2);
            Serial.print("# PREDICTION_MARGIN,");
            Serial.println((int)msg.margin);
          }

          Serial.print("# PREDICTION,");
          if (msg.prediction_state == PRED_STATIC) {
            Serial.println("static");
          } else if (msg.prediction_state == PRED_UNCERTAIN) {
            Serial.println("uncertain");
          } else {
            Serial.println(LABEL_NAMES[msg.label]);
          }

          Serial.print("# STATIC_WINDOWS,");
          Serial.println(msg.static_windows);
          Serial.print("# UNCERTAIN_WINDOWS,");
          Serial.println(msg.uncertain_windows);
          break;

        case LOG_SAMPLE_ROW:
#if ENABLE_SAMPLE_LOGGING
          Serial.print(msg.sample_id);
          Serial.print(",");
          Serial.print(msg.timestep);
          Serial.print(",");
          Serial.print(msg.t_us);
          Serial.print(",");
          Serial.print(msg.ax);
          Serial.print(",");
          Serial.print(msg.ay);
          Serial.print(",");
          Serial.print(msg.az);
          Serial.print(",");
          Serial.print(msg.gx);
          Serial.print(",");
          Serial.print(msg.gy);
          Serial.print(",");
          Serial.println(msg.gz);
#endif
          break;

        case LOG_END:
          Serial.println("# END");
          break;
      }
    }
  }
}

void enqueueLog(const LogMessage &msg) {
  if (logQueueHandle != NULL) {
    xQueueSend(logQueueHandle, &msg, portMAX_DELAY);
  }
}

void handleButtonRTOS() {
  int reading = digitalRead(BTN_PIN);

  if (reading != lastButtonReading) {
    lastDebounceMs = millis();
  }

  if ((millis() - lastDebounceMs) > DEBOUNCE_MS) {
    if (reading != buttonState) {
      buttonState = reading;

      if (buttonState == LOW && !isRecording && captureTaskHandle != NULL) {
        xTaskNotifyGive(captureTaskHandle);
      }
    }
  }

  lastButtonReading = reading;
}

void recordOneSampleRTOS() {
  isRecording = true;

  LogMessage msg = {};
  msg.type = LOG_GET_READY;
  enqueueLog(msg);

  digitalWrite(LED_PIN, HIGH);
  vTaskDelay(pdMS_TO_TICKS(1000));

  msg = {};
  msg.type = LOG_START;
  enqueueLog(msg);

  jitterMaxAbsUs = 0;
  jitterSumAbsUs = 0;
  prevSampleUs = 0;
  intervalErrorSumUs = 0;
  intervalErrorSqSumUs = 0;
  intervalCount = 0;

  accDeltaSum = 0;
  gyroAbsSum = 0;
  avgAccDelta = 0;
  avgGyroAbs = 0;
  isStaticWindow = false;

  int16_t prevAx = 0, prevAy = 0, prevAz = 0;
  bool hasPrev = false;

  const TickType_t samplePeriodTicks = pdMS_TO_TICKS(1000 / SAMPLE_RATE_HZ);
  TickType_t lastWakeTick = xTaskGetTickCount();
  unsigned long nextSampleUs = micros();

  for (int i = 0; i < NUM_SAMPLES; i++) {
    unsigned long tUs = micros();
    int16_t ax, ay, az, gx, gy, gz;

    if (i == 0) {
      totalStartUs = micros();
    }

    bool ok = readImu6_raw(&ax, &ay, &az, &gx, &gy, &gz);

    if (!ok) {
      LogMessage errMsg = {};
      errMsg.type = LOG_ERROR_STEP;
      errMsg.timestep = i;
      enqueueLog(errMsg);

      LogMessage endMsg = {};
      endMsg.type = LOG_END;
      enqueueLog(endMsg);

      digitalWrite(LED_PIN, LOW);
      isRecording = false;
      return;
    }

    // 02 + 03: immediately quantize with fixed-point preprocessing to remove
    // the 1.8 KB raw_samples[][] SRAM buffer and avoid float preprocessing.
    preprocess_one_sample_fixed(ax, ay, az, gx, gy, gz, nn_input[i]);

    // 04: motion gate features from raw IMU values.
    if (hasPrev) {
      accDeltaSum += abs32((int32_t)ax - prevAx);
      accDeltaSum += abs32((int32_t)ay - prevAy);
      accDeltaSum += abs32((int32_t)az - prevAz);
    }
    gyroAbsSum += abs32(gx);
    gyroAbsSum += abs32(gy);
    gyroAbsSum += abs32(gz);
    prevAx = ax;
    prevAy = ay;
    prevAz = az;
    hasPrev = true;

#if ENABLE_JITTER_REPORT
    long jitterUs = (long)(tUs - nextSampleUs);
    long absJitterUs = (jitterUs >= 0) ? jitterUs : -jitterUs;
    jitterSumAbsUs += absJitterUs;
    if (absJitterUs > jitterMaxAbsUs) jitterMaxAbsUs = absJitterUs;

    if (i > 0) {
      long intervalErrorUs = (long)(tUs - prevSampleUs) - (long)SAMPLE_INTERVAL_US;
      intervalErrorSumUs += intervalErrorUs;
      intervalErrorSqSumUs += (int64_t)intervalErrorUs * (int64_t)intervalErrorUs;
      intervalCount++;
    }
    prevSampleUs = tUs;
#endif

#if ENABLE_SAMPLE_LOGGING
    LogMessage sampleMsg = {};
    sampleMsg.type = LOG_SAMPLE_ROW;
    sampleMsg.sample_id = sampleId;
    sampleMsg.timestep = i;
    sampleMsg.t_us = tUs;
    sampleMsg.ax = ax;
    sampleMsg.ay = ay;
    sampleMsg.az = az;
    sampleMsg.gx = gx;
    sampleMsg.gy = gy;
    sampleMsg.gz = gz;
    enqueueLog(sampleMsg);
#endif

    nextSampleUs += SAMPLE_INTERVAL_US;

    if (i < NUM_SAMPLES - 1) {
      vTaskDelayUntil(&lastWakeTick, samplePeriodTicks);
    }
  }

#if ENABLE_JITTER_REPORT
  double intervalJitterStdUs = 0.0;
  if (intervalCount > 0) {
    double mean = (double)intervalErrorSumUs / (double)intervalCount;
    double meanSq = (double)intervalErrorSqSumUs / (double)intervalCount;
    double variance = meanSq - mean * mean;
    if (variance < 0.0) variance = 0.0;
    intervalJitterStdUs = sqrt(variance);
  }

  LogMessage jitterMsg = {};
  jitterMsg.type = LOG_JITTER_REPORT;
  jitterMsg.jitter_max_abs_us = jitterMaxAbsUs;
  jitterMsg.jitter_avg_abs_us = jitterSumAbsUs / NUM_SAMPLES;
  jitterMsg.interval_jitter_std_us = intervalJitterStdUs;
  enqueueLog(jitterMsg);
#endif

  avgAccDelta = accDeltaSum / (NUM_SAMPLES - 1);
  avgGyroAbs = gyroAbsSum / NUM_SAMPLES;
  isStaticWindow = (avgAccDelta < STATIC_ACC_DELTA_TH && avgGyroAbs < STATIC_GYRO_ABS_TH);

  LogMessage motionMsg = {};
  motionMsg.type = LOG_MOTION_REPORT;
  motionMsg.avg_acc_delta = avgAccDelta;
  motionMsg.avg_gyro_abs = avgGyroAbs;
  motionMsg.is_static = isStaticWindow;
  enqueueLog(motionMsg);

  if (inferenceTaskHandle != NULL) {
    xTaskNotifyGive(inferenceTaskHandle);
  }
}

void runInferenceRTOS() {
  unsigned long inferenceStartUs = micros();

  // 04: static windows skip smoothing, preprocess, and the whole NN stage.
  // Sampling is still needed because the motion gate uses a full input window.
  if (isStaticWindow) {
    unsigned long endUs = micros();
    staticWindowCount++;

    LogMessage resultMsg = {};
    resultMsg.type = LOG_INFERENCE_RESULT;
    resultMsg.inference_latency_us = endUs - inferenceStartUs;
    resultMsg.total_latency_us = (totalStartUs == 0) ? resultMsg.inference_latency_us : (endUs - totalStartUs);
    resultMsg.skip_nn = true;
    resultMsg.prediction_state = PRED_STATIC;
    resultMsg.static_windows = staticWindowCount;
    resultMsg.uncertain_windows = uncertainWindowCount;
    enqueueLog(resultMsg);

    LogMessage endMsg = {};
    endMsg.type = LOG_END;
    enqueueLog(endMsg);

    digitalWrite(LED_PIN, LOW);
    sampleId++;
    isRecording = false;
    return;
  }

#if ENABLE_INPUT_SMOOTHING
  // 04: moving-average smoothing before NN for moving windows only.
  smooth_quantized_samples_moving_average(nn_input);
#endif

  conv1d(&nn_input[0][0], &conv1_pool_buf[0][0], CONV1_W, CONV1_B, NN_INPUT_LEN, NN_INPUT_CH, NN_CONV1_OUT_CH, RQ_MULT_CONV1, RQ_SHIFT_CONV1);
  relu(&conv1_pool_buf[0][0], NN_INPUT_LEN * NN_CONV1_OUT_CH);

  // 02: reuse conv1 activation buffer as pooled output.
  maxpool1d_inplace_conv1(conv1_pool_buf, NN_INPUT_LEN);

  // 03: conv1d() itself is optimized in nn_ops.h.
  conv1d(&conv1_pool_buf[0][0], &conv2_out[0][0], CONV2_W, CONV2_B, NN_INPUT_LEN / 2, NN_CONV1_OUT_CH, NN_CONV2_OUT_CH, RQ_MULT_CONV2, RQ_SHIFT_CONV2);
  relu(&conv2_out[0][0], (NN_INPUT_LEN / 2) * NN_CONV2_OUT_CH);
  global_avg_pool(conv2_out, gap_out, NN_INPUT_LEN / 2);
  dense(gap_out, dense1_out, DENSE1_W, DENSE1_B, NN_CONV2_OUT_CH, NN_DENSE1_OUT, RQ_MULT_DENSE1, RQ_SHIFT_DENSE1);
  relu(dense1_out, NN_DENSE1_OUT);
  dense(dense1_out, logits, DENSE2_W, DENSE2_B, NN_DENSE1_OUT, NN_NUM_CLASSES, RQ_MULT_DENSE2, RQ_SHIFT_DENSE2);

  unsigned long endUs = micros();
  unsigned long inferenceLatencyUs = endUs - inferenceStartUs;
  unsigned long totalLatencyUs = (totalStartUs == 0) ? inferenceLatencyUs : (endUs - totalStartUs);
  int label = argmax(logits, NN_NUM_CLASSES);
  int16_t margin = prediction_margin_int16(logits, NN_NUM_CLASSES);

  int predictionState = PRED_GESTURE;
  if (margin < LOW_CONF_MARGIN_TH) {
    predictionState = PRED_UNCERTAIN;
    uncertainWindowCount++;
  }

  LogMessage resultMsg = {};
  resultMsg.type = LOG_INFERENCE_RESULT;
  resultMsg.inference_latency_us = inferenceLatencyUs;
  resultMsg.total_latency_us = totalLatencyUs;
  resultMsg.skip_nn = false;
  resultMsg.label = label;
  resultMsg.prediction_state = predictionState;
  resultMsg.logit0 = logits[0];
  resultMsg.logit1 = logits[1];
  resultMsg.logit2 = logits[2];
  resultMsg.margin = margin;
  resultMsg.static_windows = staticWindowCount;
  resultMsg.uncertain_windows = uncertainWindowCount;
  enqueueLog(resultMsg);

  LogMessage endMsg = {};
  endMsg.type = LOG_END;
  enqueueLog(endMsg);

  digitalWrite(LED_PIN, LOW);
  sampleId++;
  isRecording = false;
}
