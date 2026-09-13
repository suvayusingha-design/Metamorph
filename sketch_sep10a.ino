#define BLYNK_TEMPLATE_ID   "TMPL753433"
#define BLYNK_TEMPLATE_NAME "Quickstart Template"
#define BLYNK_AUTH_TOKEN    "Efhew2q2421I8F6y26JFWH0PqcL_J7q-"

#define BLYNK_PRINT Serial
#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>
#include "DFRobot_C4001.h" // Official DFRobot library

// 🌐 Enter your home Wi-Fi details here:
char ssid[] = "motoedge60fusion_1778";
char pass[] = "ritamsuvayu12345";

// Hardware Serial 2 pins on your ESP32
#define RXD2 16  // Connect to C4001 TX pin
#define TXD2 17  // Connect to C4001 RX pin

// Initialize the radar sensor over Hardware Serial 2 using library definitions
DFRobot_C4001_UART radar(&Serial2, 9600, RXD2, TXD2); 

BlynkTimer timer;

void checkRadarSensor() {
  // Request data updates and query target attributes using official library functions
  float distance = radar.getTargetRange(); // Correct function for distance in meters
  float speed = radar.getTargetSpeed();   // Correct function for speed in m/s
  int targetNumber = radar.getTargetNumber(); // Number of targets tracked (presence indicator)
  
  // Logic to determine presence: if targets exist or motion is tracking
  int presence = (targetNumber > 0 || distance > 0.0 || abs(speed) > 0.05) ? 1 : 0; 

  // Push data streams directly to your Blynk Cloud Datastreams
  Blynk.virtualWrite(V1, presence);
  Blynk.virtualWrite(V2, distance);
  Blynk.virtualWrite(V3, speed);
  
  // Output details onto the Arduino Serial Monitor for easy viewing
  Serial.print("Presence: "); Serial.print(presence);
  Serial.print(" | Targets: "); Serial.print(targetNumber);
  Serial.print(" | Dist: "); Serial.print(distance); Serial.print("m");
  Serial.print(" | Speed: "); Serial.print(speed); Serial.println("m/s");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  // Set up the C4001 sensor connection using official library check parameters
  while (!radar.begin()) {
    Serial.println("Sensor communication error! Verify your RX/TX pin wiring.");
    delay(1000);
  }
  Serial.println("C4001 Radar initialized successfully.");

  // Establish online cloud connection
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);

  // Set the timer loop to stream radar states once every 500ms
  timer.setInterval(500L, checkRadarSensor);
}

void loop() {
  Blynk.run();
  timer.run();
}
