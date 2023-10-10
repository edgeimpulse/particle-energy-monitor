#pragma once

#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include "UIElement.h"

class UIDisplay {
private:
    Adafruit_SH1107 _display;
    UIElement lampElement;
    UIElement blenderElement;
    UIElement fridgeElement;
    UIElement fanElement;

public:
    UIDisplay();

    void begin();
    void drawAll(int dsp_ms, int inf_ms, int ano_ms, bool anomaly);
    void setLampState(ElementState state);
    void setBlenderState(ElementState state);
    void setFridgeState(ElementState state);
    void setFanState(ElementState state);
    void debugADC(int32_t currentRaw, int32_t voltageRaw);
    void cloudIsConnecting();
    void anomalyAlert();
    void plugMeIn();
    void clearDisplay();
    void displayLogos();
};
