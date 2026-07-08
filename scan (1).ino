#include "WiFi.h"

int scan_id = 0;
const int intervalo_ms = 3000;

void setup() {
  Serial.begin(115200);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(1000);

  Serial.println("SCAN_ID,SSID,BSSID,CHANNEL,RSSI");
}

void loop() {
  scan_id++;

  int n = WiFi.scanNetworks();

  if (n == 0) {
    Serial.print(scan_id);
    Serial.println(",NO_NETWORK,NO_BSSID,0,-100");
  } else {
    for (int i = 0; i < n; i++) {
      Serial.print(scan_id);
      Serial.print(",");

      Serial.print(WiFi.SSID(i));
      Serial.print(",");

      Serial.print(WiFi.BSSIDstr(i));
      Serial.print(",");

      Serial.print(WiFi.channel(i));
      Serial.print(",");

      Serial.println(WiFi.RSSI(i));
    }
  }

  Serial.println("END_SCAN");

  WiFi.scanDelete();

  delay(intervalo_ms);
}