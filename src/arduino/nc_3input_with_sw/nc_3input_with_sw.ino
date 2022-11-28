// Nail conductor 3input with 1switch.

int pulsePin1 = 0;                // Pulse Sensor analog pin
int pulsePin2 = 1;                // Pulse Sensor analog pin
int pulsePin3 = 2;                // Pulse Sensor analog pin

int switchPin1 = 2;               // used input switch pin

int pwmPin = 9;  // PWM出力させるピン番号を指定

volatile int Switch_push;
volatile int Signal1;             // holds the incoming raw data
volatile int Signal2;             // holds the incoming raw data
volatile int Signal3;             // holds the incoming raw data

int DUTY = 255 * 1.0;   // デューティー比を指定(0~255)
int DELAY = 10;

void setup() {
  // No. 9/10 Pin ベースクロック変更
  //  TCCR1B = (TCCR1B & 0b11111000) | 0x01; //31.373kHz
  TCCR1B = (TCCR1B & 0b11111000) | 0x02; // 3.921kHz
  //  TCCR1B = (TCCR1B & 0b11111000) | 0x03; //   490.2Hz
  //  TCCR1B = (TCCR1B & 0b11111000) | 0x04; //   122.6Hz
  //  TCCR1B = (TCCR1B & 0b11111000) | 0x05; //    30.6kHz

  pinMode(switchPin1, INPUT_PULLUP);
  pinMode(pwmPin, OUTPUT);  // PWM出力用設定

  Serial.begin(115200);
  while (!Serial);
  //  Serial.println("Ready!");
}

void loop() {
  Switch_push = !digitalRead(switchPin1); // Switchが押されたか

  analogWrite(pwmPin, DUTY);  // 指定ピンにアナログ値(PWM)を出力

  Signal1 = Signal2 = Signal3 = 0;

  Signal1 = analogRead(pulsePin1);// / 1.024;      // read the Pulse Sensor
  Signal2 = analogRead(pulsePin2);// / 1.024;      // read the Pulse Sensor
  Signal3 = analogRead(pulsePin3);// / 1.024;      // read the Pulse Sensor

  serialOutput();
  delay(DELAY);
}

void serialOutput() {  // Output Serial Raw data.
  Serial.print(Signal1);
  Serial.print(",");
  Serial.print(Signal2);
  Serial.print(",");
  Serial.print(Signal3);
  Serial.print(",");
  Serial.println(Switch_push);
}
