/*
 * Copyright (c) 2023 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS
 * IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * This code implements standalone inferencing of this publicly available project:
 * https://studio.edgeimpulse.com/public/288532/latest
 * 
 * Particle & Edge Impulse Energy Monitor featured at Imagine 2023
 */

/* Include ----------------------------------------------------------------- */
#include <Imagine-Demo-All-Permutations_inferencing.h>
#include "Particle.h"
#include "UIDisplay.h"
#include <Wire.h>

#define NUM_SAMPLES     40  //number of samples per event
#define GAIN_ALL        245         //245 is the raw gain setting used such that when all demo devices are on there's no clipping
#define GAIN_VALUE      GAIN_ALL    //212 is default gain for 15A full scale
#define CLOUD_TIMEOUT   60000       //time in ms to wait for cloud connect before running inferencing
#define AC_ON_THRESHOLD 3000        //min raw ADC read of voltage to count as being plugged in

void sampleData();
void dataCakeSend();

int payloadBuffer[2][NUM_SAMPLES];
int sampleIndex = 0;
char buf[1024];
bool sendData = false;
bool sendCake = false;
bool firstPowerOn = true;
int32_t voltageADC = 0; //init voltage value to zero
int32_t voltageMAX = 0;
int32_t currentADC = 0; //init current value to zero
int voltagePIN = D13; //voltage is on A2
int currentPIN = A5; //current is pn A5

//power measurement
double filteredCurrent = 0;
double offsetCurrent = 0;
float sumCurrent = 0;
double squareCurrent = 0;
float rmsCurrent = 0;
float instPower = 0;
double CURRENT_CAL = 0.0027566;

//Sampling interval is 1 ms -> 1 kHz sampling frequency
Timer sample(1, sampleData);

//Timer used to send states and (uncalibrated) instant power to cloud
Timer dataCake(5000, dataCakeSend);

//Create ui 
UIDisplay ui;

//Red LED on custom analog front end
int redLED = D3;

// Classifier lookup value for switch case & thresholds
int classifierLookup;
float confidenceThreshold = 0.5;
float anomalyThreshold = 90.0;

// Appliance & anomaly ON/OFF states
bool lampState = false;
bool fanState = false;
bool fridgeState = false;
bool blenderState = false;
bool anomalyState = false;

void sampleData()
{
    voltageADC = analogRead(voltagePIN); //read ADC values
    currentADC = analogRead(currentPIN);
    payloadBuffer[0][sampleIndex] = voltageADC;
    payloadBuffer[1][sampleIndex] = currentADC;
    sampleIndex++;

    filteredCurrent = currentADC - 2048;
    squareCurrent = filteredCurrent * filteredCurrent;
    sumCurrent += squareCurrent;

    if(sampleIndex == NUM_SAMPLES)  //we are done sampling for this event
    {
        sendData = true;
        sampleIndex = 0;
        sample.stop();  
        rmsCurrent = sqrt(sumCurrent / NUM_SAMPLES);
        rmsCurrent -= 8;
        if(rmsCurrent < 0)
        {
            rmsCurrent = 0;
        }
        rmsCurrent *= CURRENT_CAL;
        instPower = 120 * rmsCurrent;   //For simplicity we use 120 for RMS voltage as if everything is at power factor of 1
        sumCurrent = 0;                 //This is not really the case but accurate power metrics can be calculated on-device if desired 
    }
}

void dataCakeSend()
{
    sendCake = true;
}

SYSTEM_THREAD(ENABLED);

SerialLogHandler logHandler(LOG_LEVEL_ERROR);

/* Forward declerations ---------------------------------------------------- */
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr);
void setup();
void loop();
void updateStates(ei_impulse_result_t result);
void updateLCD(int dspTime, int infTime, int anoTime);
bool isPlugged(void);

static const float features[] = {1435, 1908, 1037, 1739, 712, 1513, 588, 1248, 609, 1273, 812, 1369, 1499, 1693, 2110, 1937, 2550, 2124, 2935, 2273, 3314, 2479, 3501, 2782, 3519, 2897, 3428, 2768, 2839, 2507, 2175, 2201, 1709, 2016, 1321, 1877, 888, 1660, 650, 1396, 585, 1197, 634, 1290, 1000, 1458, 1713, 1783, 2277, 2015, 2665, 2166, 3073, 2303, 3400, 2598, 3525, 2867, 3498, 2808, 3310, 2733, 2615, 2403, 1990, 2131, 1569, 1973, 1166, 1834, 797, 1602, 596, 1288, 591, 1240, 689, 1307, 1248, 1562};

void setGain(void)
{
    char gainRet = 0;
    Wire.begin();
    delay(5);
    Wire.beginTransmission(0x2f);
    Wire.write(0x00);
    Wire.write(GAIN_VALUE);
    Wire.endTransmission();
    delay(10);
    Wire.requestFrom(0x2f,1);
    gainRet = Wire.read();
    Wire.endTransmission(); 
    if(gainRet == GAIN_VALUE)
    {
         ei_printf("SUCCESS SETTING GAIN: %d", gainRet);
    }
    else
    {
        ei_printf("FAILED TO SET GAIN: %d", gainRet);
    }  
}



/**
 * @brief      Copy raw feature data in out_ptr
 *             Function called by inference library
 *
 * @param[in]  offset   The offset
 * @param[in]  length   The length
 * @param      out_ptr  The out pointer
 *
 * @return     0
 */
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    int out_ptr_ix = 0;
    sample.start();
    while(sendData == false);
    for(int j=0; j<40; j++)
    {
        out_ptr[out_ptr_ix++] = (float)(payloadBuffer[0][j]);
        out_ptr[out_ptr_ix++] = (float)(payloadBuffer[1][j]);
    }
    return 0;
}

void print_inference_result(ei_impulse_result_t result);


/**
 * @brief      Particle setup function
 */
void setup()
{
    // put your setup code here, to run once:
    pinMode(redLED, OUTPUT);
    digitalWrite(redLED, LOW);
    // Wait for serial to make it easier to see the serial logs at startup.
    waitFor(Serial.isConnected, 1000);
    ui.begin();
    delay(2000);
    ei_printf("Edge Impulse Energy Monitor for Particle Devices\r\n");
    delay(10);
    setGain();
    delay(3000);
    Particle.connect();
    delay(100);
    ui.clearDisplay();
}

/**
 * @brief      Particle main function
 */
void loop()
{
    static unsigned long now = millis();

    //Only if we are actually plugged into an outlet
    if(isPlugged())
    {
        now = millis();
        if(firstPowerOn)    //First loop requires cloud timeout
        {
            while(millis() - now < CLOUD_TIMEOUT)
            {
                digitalWrite(redLED, LOW);
                delay(500);
                ui.cloudIsConnecting();
                digitalWrite(redLED, HIGH);
                if(Particle.connected() == true)
                {
                    dataCake.start();
                    break;
                }
            }
            if(Particle.connected() == false)
            {
                Particle.disconnect();  //Don't keep trying to connect if we haven't during timeout
            }
            firstPowerOn = false;
        }

        now = millis();

        ei_impulse_result_t result = { 0 };

        // the features are stored into flash, and we don't want to load everything into RAM
        signal_t features_signal;
        features_signal.total_length = 80;
        features_signal.get_data = &raw_feature_get_data;

        // invoke the impulse
        EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);
        if (res != EI_IMPULSE_OK) {
            ei_printf("ERR: Failed to run classifier (%d)\n", res);
            return;
        }
        updateStates(result);
        updateLCD(result.timing.dsp, result.timing.classification, result.timing.anomaly);

        if(Particle.connected() && sendCake)
        {
            memset(buf, 0, sizeof(buf));
            JSONBufferWriter writer(buf, sizeof(buf) - 1);
            writer.beginObject();
                writer.name("blender").value(blenderState);
                writer.name("fridge").value(fridgeState);
                writer.name("fan").value(fanState);
                writer.name("lamp").value(lampState);
                writer.name("anomaly").value(anomalyState);
                writer.name("power").value(instPower);
            writer.endObject();
            Particle.publish("datacake/states",buf);
            sendCake = false;
        }

        // print inference return code
        ei_printf("run_classifier returned: %d\r\n", res);
        print_inference_result(result);
    }// end if plugged in
    else
    {
        ui.plugMeIn();  // Indicate to user to plug in board
        ei_printf("PLUG ME IN!\r\n");
        delay(1000);
    }

}

void print_inference_result(ei_impulse_result_t result) {

    // Print how long it took to perform inference
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
            result.timing.dsp,
            result.timing.classification,
            result.timing.anomaly);

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }

    // Print the prediction results (classification)
#else
    ei_printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
        ei_printf("%.5f\r\n", result.classification[i].value);
    }
#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

}

void updateStates(ei_impulse_result_t result)
{
    static int classifierPast = 255;
    static int classifierChange = 0;
    static int anomalyChange = 0;
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if(result.classification[i].value > confidenceThreshold){
            if(i != classifierPast) // state change
            {
                classifierChange++;
                if(classifierChange > 1)    // are you sure about that??
                {
                    classifierLookup = i;   // actually change state
                    classifierPast = classifierLookup;
                    classifierChange = 0;
                }
            }
            else
            {
                classifierChange = 0;
            }
        }
    }
    
    if(result.anomaly > anomalyThreshold)
    {   
        if(classifierLookup < 8)    // if blender filter anomaly results
        {
            if(anomalyChange++ > 2)
            {
                anomalyState = true;
                anomalyChange = 0;
            }
        }
        else                        // otherwise if not in blender state filter anomaly less
        {
            if(anomalyChange++ > 1)
            {
                anomalyState = true;
                anomalyChange = 0;
            }
        }
    }
    else
    {
        anomalyState = false;
    }
    switch (classifierLookup)
    {
    case 0:
        blenderState = true;
        fanState = false;
        fridgeState = false;
        lampState = false;
        break;
    case 1:
        blenderState = true;
        fanState = true;
        fridgeState = false;
        lampState = false;
        break;
    case 2:
        blenderState = true;
        fanState = true;
        fridgeState = false;
        lampState = true;
        break;
    case 3:
        blenderState = true;
        fanState = false;
        fridgeState = true;
        lampState = false;
        break;
    case 4:
        blenderState = true;
        fanState = true;
        fridgeState = true;
        lampState = false;
        break;
    case 5:
        blenderState = true;
        fanState = true;
        fridgeState = true;
        lampState = true;
        break;
    case 6:
        blenderState = true;
        fanState = false;
        fridgeState = true;
        lampState = true;
        break;
    case 7:
        blenderState = true;
        fanState = false;
        fridgeState = false;
        lampState = true;
        break;
    case 8:
        blenderState = false;
        fanState = true;
        fridgeState = false;
        lampState = false;
        break;
    case 9:
        blenderState = false;
        fanState = false;
        fridgeState = true;
        lampState = false;
        break;
    case 10:
        blenderState = false;
        fanState = true;
        fridgeState = true;
        lampState = false;
        break;
    case 11:
        blenderState = false;
        fanState = false;
        fridgeState = false;
        lampState = true;
        break;
    case 12:
        blenderState = false;
        fanState = true;
        fridgeState = false;
        lampState = true;
        break;
    case 13:
        blenderState = false;
        fanState = false;
        fridgeState = true;
        lampState = true;
        break;
    case 14:
        blenderState = false;
        fanState = true;
        fridgeState = true;
        lampState = true;
        break;
    case 15:
        blenderState = false;
        fanState = false;
        fridgeState = false;
        lampState = false;
        break;      
    default:
        blenderState = false;
        fanState = false;
        fridgeState = false;
        lampState = false;
        break;
    }
}

void updateLCD(int dspTime, int infTime, int anoTime){
    static bool anomalyTrig = false;
    static bool anomalyGone = true;
    if(anomalyState)
    {
        anomalyTrig = true;
        anomalyGone = false;
        ui.anomalyAlert();
    }
    else
    {
        if(anomalyTrig) //anomalyState went low after going high
        {
            anomalyGone = true;
            anomalyTrig = false;
        }
        if(blenderState)
        {
            ui.setBlenderState(ON);
        }
        else if(!blenderState)
        {
            ui.setBlenderState(OFF);
        }
        if(fanState)
        {
            ui.setFanState(ON);
        }
        else if(!fanState)
        {
            ui.setFanState(OFF);
        }
        if(fridgeState)
        {
            ui.setFridgeState(ON);
        }
        else if(!fridgeState)
        {
            ui.setFridgeState(OFF);
        }
        if(lampState)
        {
            ui.setLampState(ON);
        }
        else if(!lampState)
        {
            ui.setLampState(OFF);
        }

        if(anomalyGone)
        {
            ui.clearDisplay();
            anomalyGone = false;
        }
        ui.drawAll(dspTime, infTime, anoTime, anomalyState);
    }//end if not anomaly
}

bool isPlugged(void)
{
    bool pluggedIn;
    unsigned long now = millis();
    while(millis() - now < 19)     // capture max reading during slightly longer than 1 period at 60 Hz to detect presence of AC
    {
        voltageADC = analogRead(voltagePIN);
        if(voltageADC > voltageMAX)
        {
            voltageMAX = voltageADC;    //update max raw voltage read
        }
    }
    if(voltageMAX > AC_ON_THRESHOLD)
    {
        pluggedIn = true;
    }
    else
    {
        pluggedIn = false;
    }
    voltageMAX = 0;
    return pluggedIn;
}