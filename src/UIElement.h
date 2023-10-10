#pragma once

#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>

enum ElementState {
    ON,
    OFF
};

class UIElement {
private:
    const uint8_t *icon;
    int posX, posY;

    const unsigned int WIDTH = 64;
    const unsigned int ICON_SIZE = 24;
    const unsigned int TEXT_ON_X = 34;
    const unsigned int TEXT_OFF_X = 27;
    const unsigned int TEXT_Y = 17;

    ElementState state;

public:
    UIElement(const uint8_t *iconBitmap, int x, int y);

    ElementState getState() const;
    void setState(ElementState newState);
    void draw(Adafruit_SH1107& display);
};
