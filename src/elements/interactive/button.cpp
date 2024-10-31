#include <string>

#include <SFML/Graphics.hpp>

#include "button.hpp"
#include "config.hpp"

Button::Button(
    float x,
    float y,
    const std::string& _text,
    const std::string& font_path,
    sf::Color main_color
) {
    font.loadFromFile(font_path);
    text.setPosition(x + padding, y + padding);
    text.setFont(font);
    text.setString(_text);
    text.setCharacterSize(24);
    text.setFillColor(main_color);
    sf::FloatRect text_bounding = text.getGlobalBounds();

    shape.setPosition(x, y);
    shape.setOutlineColor(main_color);
    shape.setOutlineThickness(2);
    shape.setSize(sf::Vector2f(
        text_bounding.width + padding*2,
        text_bounding.height + padding*3
    ));
    recalculateColor();
}

Button::~Button() {
    delete current_color;
}

void Button::handleEvent(const sf::Event& event) {
    if (event.type != sf::Event::MouseMoved) return;
    switch (event.type) {
        case sf::Event::MouseMoved: {
            hovering = isMouseOver();
            recalculateColor();
        }
    }
}

void Button::draw(sf::RenderWindow &window) {
    // std::cout << "Color set: ["
    //     << (int) current_color->r << ", "
    //     << (int) current_color->g << ", "
    //     << (int) current_color->b << ", "
    //     << (int) current_color->a << ", "
    //     << "], active: " << active << std::endl;
    shape.setFillColor(*current_color);
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

void Button::recalculateColor() {
    delete current_color;
    current_color = new sf::Color(255, 255, 255, 0);
    if (hovering) *current_color += hover_color;
    if (active) *current_color += active_color;
}

void Button::setActive(bool _active) {
    active = _active;
    recalculateColor();
}

void Button::setHovering(bool _hovering) {
    hovering = _hovering;
    recalculateColor();
}

const sf::Color Button::hover_color = sf::Color(255, 255, 255, 60);
const sf::Color Button::active_color = sf::Color(255, 255, 255, 60);
