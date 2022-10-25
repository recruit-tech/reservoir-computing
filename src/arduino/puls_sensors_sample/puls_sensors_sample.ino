
/*  PulseSensor Starter Project and Signal Tester
 *  The Best Way to Get Started  With, or See the Raw Signal of, your PulseSensor.com邃｢ & Arduino.
 *
 *  Here is a link to the tutorial
 *  https://pulsesensor.com/pages/code-and-guide
 *
 *  WATCH ME (Tutorial Video):
 *  https://www.youtube.com/watch?v=RbB8NSRa5X4
 *
 *
-------------------------------------------------------------
1) This shows a live human Heartbeat Pulse.
2) Live visualization in Arduino's Cool "Serial Plotter".
3) Blink an LED on each Heartbeat.
4) This is the direct Pulse Sensor's Signal.
5) A great first-step in troubleshooting your circuit and connections.
6) "Human-readable" code that is newbie friendly."

*/
//#include <HardwareBLESerial.h>

//HardwareBLESerial &bleSerial = HardwareBLESerial::getInstance();

unsigned char BUTTON;
#define BUTTON_001 0001
#define BUTTON_002 0001 << 1
#define BUTTON_003 0001 << 2
#define BUTTON_004 0001 << 3

//  Variables
int PulseSensorPurplePin0 = 0;        // Pulse Sensor PURPLE WIRE connected to ANALOG PIN 0
int PulseSensorPurplePin1 = 1;        // Pulse Sensor PURPLE WIRE connected to ANALOG PIN 0
int PulseSensorPurplePin2 = 2;        // Pulse Sensor PURPLE WIRE connected to ANALOG PIN 0
int PulseSensorPurplePin3 = 3;        // Pulse Sensor PURPLE WIRE connected to ANALOG PIN 0
int LED13 = 13;   //  The on-board Arduion LED


int Signal_000;                // holds the incoming raw data. Signal value can range from 0-1024
int Signal_001;                // holds the incoming raw data. Signal value can range from 0-1024
int Signal_002;                // holds the incoming raw data. Signal value can range from 0-1024
int Signal_003;                // holds the incoming raw data. Signal value can range from 0-1024

//int Threshold = 550;            // Determine which Signal to "count as a beat", and which to ingore.

int BUTTON01 = 7;
int BUTTON02 = 8;
int BUTTON03 = 9;
int BUTTON04 = 10;

bool use_bluetooth = false; 

// The SetUp Function:
void setup() {
  //pinMode(LED13,OUTPUT);         // pin that will blink to your heartbeat!
  pinMode(BUTTON01, INPUT);
  pinMode(BUTTON02, INPUT);
  pinMode(BUTTON03, INPUT);
  pinMode(BUTTON04, INPUT);
  Serial.begin(115200);         // Set's up Serial Communication at certain speed.
  while (!Serial) {
    ; // シリアルポートの準備ができるのを待つ(Leonardoのみ必要)
  }
  Serial.println("Ready");
}
int i =0;
unsigned long start_time = micros();
// The Main Loop Function
void loop() {

  BUTTON = 0;
  //bleSerial.poll();
  Signal_000 = analogRead(PulseSensorPurplePin0);
  Signal_001 = analogRead(PulseSensorPurplePin1);
  Signal_002 = analogRead(PulseSensorPurplePin2);
  Signal_003 = analogRead(PulseSensorPurplePin3);
  if(digitalRead(BUTTON01) == HIGH)
  {
    BUTTON |= BUTTON_001;
  }
                                                
  if(digitalRead(BUTTON02) == HIGH)
  {
    BUTTON |= BUTTON_002;
  }
    
  if(digitalRead(BUTTON03) == HIGH)
  {
    BUTTON |= BUTTON_003;
  }

  if(digitalRead(BUTTON04) == HIGH)
  {
    BUTTON |= BUTTON_004;
  }


  unsigned long end_time = micros();
  unsigned long delta = end_time - start_time;
  char send_data[30];
  memset(&send_data,0x00,30);

  
  sprintf(send_data, "%d,%d,%d,%d,%d",Signal_000, Signal_001, Signal_002, Signal_003, BUTTON);


  if (10*1000>delta)
  {
    delayMicroseconds(10*1000-delta);
  }

     Serial.println(send_data);

  
  start_time = micros();
  
}
