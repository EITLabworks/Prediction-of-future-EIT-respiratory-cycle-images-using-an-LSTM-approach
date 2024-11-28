int analogPin = 3;


int SEND = 7; // send signal
int ECHO = 6; // echo signal

float distance; // distance
char userInput; // serial input

void setup(){
  pinMode(SEND, OUTPUT);
  pinMode(ECHO, INPUT);
  Serial.begin(9600); // setup serial

}

void loop(){

if(Serial.available()> 0){ 
    userInput = Serial.read(); // read serial connection
      if(userInput == 'g'){ // if we get expected value 'g', proceed...
        digitalWrite(SEND, HIGH); // start ultrasonic impulse
        delayMicroseconds(10); // wait 10us
        digitalWrite(SEND, LOW); // stop ultrasonic impulse
        long timeing = pulseIn(ECHO, HIGH); // get pulse time
        distance= (timeing / 2) * 0.03432; // compute distance in cm
        Serial.println(distance); // write distance to serial port
        digitalWrite(SEND, LOW); // turn off ultrasonic sensor
      }
  }
}
