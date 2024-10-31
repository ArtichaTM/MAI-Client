#include <string>
#include <iostream>

#include <SFML/Graphics.hpp>

#include "button.hpp"
#include "config.hpp"

Button::Button(
    float x,
    float y,
    const std::string& _text,
    const std::string& font_path,
    sf::Color color
) {
    font.loadFromFile(font_path);
    text.setPosition(x + padding, y + padding);
    text.setFont(font);
    text.setString(_text);
    text.setCharacterSize(24);
    text.setFillColor(color);
    sf::FloatRect text_bounding = text.getGlobalBounds();

    shape.setPosition(x, y);
    shape.setOutlineColor(color);
    shape.setOutlineThickness(2);
    shape.setFillColor(sf::Color::Transparent);
    shape.setSize(sf::Vector2f(
        text_bounding.left + text_bounding.width + padding*2,
        text_bounding.top + text_bounding.height + padding*2
    ));
}

void Button::handleEvent(const sf::Event& event) {
    if (event.type != sf::Event::MouseMoved) return;
    sf::Color color(255, 255, 255, 100);
    if (isMouseOver()) {
        shape.setFillColor(hover_color);
    } else {
        shape.setFillColor(sf::Color::Transparent);
    }
}

void Button::draw(sf::RenderWindow &window) const {
    window.draw(shape);
    window.draw(text);
}

bool Button::isMouseOver() {
    return shape.getGlobalBounds().contains(
        ROOT_WINDOW->mapPixelToCoords(
            (sf::Mouse::getPosition(*ROOT_WINDOW))
        )
    );
}

const sf::Color Button::hover_color = sf::Color(255, 255, 255, 120);

// bool Button::isMouseOver(sf::Vector2i mousePos) {
//     return buttonShape.getGlobalBounds().contains(static_cast<sf::Vector2f>(mousePos));
// }

// void Button::onClick() {

// }

// void Button::update(sf::Vector2i mousePos) {
//     if (isMouseOver(mousePos)) {
//         buttonShape.setFillColor(sf::Color::Yellow);
//     } else {
//         buttonShape.setFillColor(sf::Color::Green);
//     }
// }
