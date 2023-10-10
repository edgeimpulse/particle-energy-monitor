#include "UIElement.h"
#include "Org_01.h"

UIElement::UIElement(const uint8_t *iconBitmap, int x, int y) 
    : icon(iconBitmap), posX(x), posY(y), state(OFF) {}

ElementState UIElement::getState() const {
    return state;
}

void UIElement::setState(ElementState newState) {
    state = newState;
}

void UIElement::draw(Adafruit_SH1107& display) {
    display.setFont(&Org_01);
    display.setTextSize(2);

    if (state == ON) {
        display.fillRoundRect(posX, posY, WIDTH, ICON_SIZE + 2, 3, 1);
        display.drawBitmap(posX + 1, posY + 1, icon, ICON_SIZE, ICON_SIZE, 0);
        display.setCursor(posX + TEXT_ON_X, posY + TEXT_Y);
        display.setTextColor(0);
        display.println("ON");
    } else {
        display.fillRoundRect(posX, posY, WIDTH, ICON_SIZE + 1, 3, 0);  // Clear screen first
        
        display.drawRoundRect(posX, posY, WIDTH, ICON_SIZE + 2, 3, 1);
        display.drawBitmap(posX + 1, posY + 1, icon, ICON_SIZE, ICON_SIZE, 1);
        display.setCursor(posX + TEXT_OFF_X, posY + TEXT_Y);
        display.setTextColor(1);
        display.println("OFF");
    }

    display.display();
}
