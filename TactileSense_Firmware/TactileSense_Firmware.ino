#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <driver/i2s.h>

// TensorFlow Lite Micro Includes
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Compiled quantized model header array
#include "model.h"

// -------------------------------------------------------------
// Pin Configurations (Matching Diagram 4)
// -------------------------------------------------------------
#define I2S_WS         10    // LRCLK / WS pin
#define I2S_SD         11    // DIN / SD Data pin
#define I2S_SCK        12    // BCLK / SCK Clock pin
#define I2S_PORT       I2S_NUM_0

#define OLED_SDA       8     // I2C Data pin
#define OLED_SCL       9     // I2C Clock pin
#define SCREEN_WIDTH   128
#define SCREEN_HEIGHT  64

#define MOTOR_PIN      4     // PWM Trigger pin to Transistor Base

// Audio Configuration
#define SAMPLE_RATE    16000
#define INPUT_SAMPLES  128   // Feature length per inference frame

// -------------------------------------------------------------
// Dataset Output Classes (15 Classes Total)
// -------------------------------------------------------------
#define NUM_CLASSES 15

// Alphabetical mapping matching Python LabelEncoder
const char* LABELS[NUM_CLASSES] = {
    "Car Horn",
    "Clapping",
    "Clock Alarm",
    "Crackling Fire",
    "Crying Baby",
    "Door Knocks",
    "Fire Work",
    "Glass Breaking",
    "Rain",
    "Siren",
    "Thunderstorm",
    "Washing Machine",
    "Water Tap",
    "Background",
    "Unknown"
};

// -------------------------------------------------------------
// Peripherals & ML Globals
// -------------------------------------------------------------
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

constexpr int kTensorArenaSize = 30 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;
tflite::AllOpsResolver resolver;

// -------------------------------------------------------------
// Initialization Routines
// -------------------------------------------------------------
void setupI2S() {
  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 4,
      .dma_buf_len = 512,
      .use_apll = false
  };

  i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_SCK,
      .ws_io_num = I2S_WS,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
}

void setupOLED() {
  Wire.begin(OLED_SDA, OLED_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 OLED initialization failed!"));
    for (;;);
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 10);
  display.println("TactileSense Ready");
  display.println("Loading 15-Class AI...");
  display.display();
}

void setupTFLite() {
  tflite_model = tflite::GetModel(g_model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLite schema mismatch!");
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, nullptr);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
}

// -------------------------------------------------------------
// Categorized Haptic Vibration Feedback
// -------------------------------------------------------------
void triggerHapticFeedback(int labelIndex) {
  // Category 1: Emergency Hazards -> Urgent Double Pulse
  // (Clock Alarm, Crackling Fire, Fire Work, Glass Breaking, Siren)
  if (labelIndex == 2 || labelIndex == 3 || labelIndex == 6 || labelIndex == 7 || labelIndex == 9) {
    digitalWrite(MOTOR_PIN, HIGH);
    delay(200);
    digitalWrite(MOTOR_PIN, LOW);
    delay(100);
    digitalWrite(MOTOR_PIN, HIGH);
    delay(200);
    digitalWrite(MOTOR_PIN, LOW);
  } 
  // Category 2: Alert Events -> Single Long Pulse
  // (Car Horn, Crying Baby, Door Knocks)
  else if (labelIndex == 0 || labelIndex == 4 || labelIndex == 5) {
    digitalWrite(MOTOR_PIN, HIGH);
    delay(400);
    digitalWrite(MOTOR_PIN, LOW);
  }
  // Category 3: General Ambient Sounds -> Short Single Pulse
  // (Clapping, Rain, Thunderstorm, Washing Machine, Water Tap)
  else if (labelIndex == 1 || labelIndex == 8 || labelIndex == 10 || labelIndex == 11 || labelIndex == 12) {
    digitalWrite(MOTOR_PIN, HIGH);
    delay(150);
    digitalWrite(MOTOR_PIN, LOW);
  }
}

// -------------------------------------------------------------
// Setup & Main Processing Loop
// -------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);

  setupOLED();
  setupI2S();
  setupTFLite();

  display.clearDisplay();
  display.setCursor(0, 20);
  display.println("TactileSense Active");
  display.display();
  delay(1000);
}

void loop() {
  int16_t raw_audio_buffer[INPUT_SAMPLES];
  size_t bytes_read = 0;

  // Read I2S audio stream from INMP441
  i2s_read(I2S_PORT, &raw_audio_buffer, sizeof(raw_audio_buffer), &bytes_read, portMAX_DELAY);

  // Normalize audio data into input tensor
  for (int i = 0; i < INPUT_SAMPLES; i++) {
    input_tensor->data.f[i] = (float)raw_audio_buffer[i] / 32768.0f;
  }

  // Execute inference
  unsigned long start_time = millis();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long latency = millis() - start_time;

  if (invoke_status != kTfLiteOk) {
    Serial.println("Inference execution failed!");
    return;
  }

  // Find top predicted class across all 15 classes
  int best_class = 0;
  float max_score = 0.0f;
  for (int i = 0; i < NUM_CLASSES; i++) {
    float score = output_tensor->data.f[i];
    if (score > max_score) {
      max_score = score;
      best_class = i;
    }
  }

  float confidence = max_score * 100.0f;

  // Render on OLED display
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("TactileSense Alert");
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);

  display.setTextSize(2);
  display.setCursor(0, 20);
  display.println(LABELS[best_class]);

  display.setTextSize(1);
  display.setCursor(0, 48);
  display.print("Conf: ");
  display.print(confidence, 1);
  display.print("% | ");
  display.print(latency);
  display.print("ms");
  display.display();

  // Trigger motor for valid sounds exceeding 70% confidence
  if (confidence > 70.0f && best_class != 13) { // Exclude Background (Index 13)
    triggerHapticFeedback(best_class);
  }

  delay(200);
}