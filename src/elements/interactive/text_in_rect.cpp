#include <string>
#include <sstream>
#include <iostream>

#include <SFML/Graphics.hpp>

#include "config.hpp"
#include "./text_in_rect.hpp"
#include "text_in_rect.hpp"

TextInRect::TextInRect(
    float x,
    float y,
    const std::string& _text,
    const std::string& font_path,
    sf::Color main_color
) {
    font.loadFromFile(font_path);
    text.setFont(font);
    text.setString(_text);
    text.setCharacterSize(24);
    text.setFillColor(main_color);
    sf::FloatRect text_bounding = text.getGlobalBounds();

    shape.setPosition(x, y);
    shape.setOutlineColor(main_color);
    shape.setOutlineThickness(2);
    fit();
    recalculateColor();
}

TextInRect::~TextInRect() { delete current_color; }

TextInRect::operator std::string() const {
    std::stringstream ss;
    ss << "<TextInRect with text \"" << (std::string) text.getString() << "\">";
    return ss.str();
}

void TextInRect::draw(sf::RenderTarget &target, sf::RenderStates states) const {
    target.draw(shape);
    target.draw(text);
}

sf::FloatRect TextInRect::getGlobalBounds() const { return shape.getGlobalBounds(); }

void TextInRect::move(const float left, const float top)
{
    sf::Vector2f shape_position(shape.getPosition());
    sf::Vector2f text_position(text.getPosition()); 
    shape.setPosition(
        shape_position.x + left,
        shape_position.y + top
    );
    text.setPosition(
        text_position.x + left,
        text_position.y + top
    );
}

void TextInRect::fit() {
    sf::FloatRect text_bounding = text.getGlobalBounds();
    sf::FloatRect shape_bounding = shape.getGlobalBounds();
    text.setPosition(
        shape_bounding.left + padding,
        shape_bounding.top + padding
    );
    shape.setSize(sf::Vector2f(
        text_bounding.width + padding*2 + 2.f,
        text_bounding.height + padding*3 + 3.f
    ));
}

TextInRect* TextInRect::setPadding(float padding) {
    this->padding = padding;
    return this;
}

void TextInRect::recalculateColor() {
    delete current_color;
    current_color = new sf::Color(255, 255, 255, 0);
    if (hovering) *current_color += hover_color;
    if (active) *current_color += active_color;
    shape.setFillColor(*current_color);
}

bool TextInRect::getActive() { return active; }
bool TextInRect::getHovering() { return hovering; }

void TextInRect::setActive(bool _active) {
    if (_active == active) return;
    active = _active;
    recalculateColor();
}

void TextInRect::setHovering(bool _hovering) {
    if (_hovering == hovering) return;
    hovering = _hovering;
    recalculateColor();
}

void TextInRect::keyPressed(const sf::Mouse::Button& button) {
    setActive(button == sf::Mouse::Button::Left);
}

void TextInRect::mouseMoved(const sf::Vector2f& vector) {
    setHovering(isVectorInBounds(vector));
}

const sf::Color TextInRect::hover_color = sf::Color(255, 255, 255, 60);
const sf::Color TextInRect::active_color = sf::Color(255, 255, 255, 60);
