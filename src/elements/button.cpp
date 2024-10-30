#include <string>
#include <SFML/Graphics.hpp>

#include "button.hpp"

Button::Button(
    float x,
    float y,
    float width,
    float height,
    const std::string& text
) {
    buttonShape.setSize(sf::Vector2f(width, height));
    buttonShape.setPosition(x, y);
    buttonShape.setFillColor(sf::Color::Green);

    buttonFont.loadFromFile("../static/fonts/microsoftsansserif.ttf");
    buttonText.setFont(buttonFont);
    buttonText.setString(text);
    buttonText.setCharacterSize(24);
    buttonText.setFillColor(sf::Color::White);
    buttonText.setPosition(
        x + (width - buttonText.getGlobalBounds().width) / 2,
        y + (height - buttonText.getGlobalBounds().height) / 2
    );
}

void Button::draw(sf::RenderWindow& window) {
    window.draw(buttonShape);
    window.draw(buttonText);
}

bool Button::isMouseOver(sf::Vector2i mousePos) {
    return buttonShape.getGlobalBounds().contains(static_cast<sf::Vector2f>(mousePos));
}

void Button::onClick() {}

void Button::update(sf::Vector2i mousePos) {
    if (isMouseOver(mousePos)) {
        buttonShape.setFillColor(sf::Color::Yellow);
    } else {
        buttonShape.setFillColor(sf::Color::Green);
    }
}
