#include <string>
#include <sstream>
#include <iostream>

#include <SFML/Graphics.hpp>

#include "config.hpp"
#include "./text_in_rect.hpp"

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

TextInRect* TextInRect::setLeft(float value) {
    shape.setPosition(value, getTop());
    text.setPosition(value, getTop());
    return this;
}
TextInRect* TextInRect::setTop(float value) {
    shape.setPosition(getLeft(), value);
    text.setPosition(getLeft(), value);
    return this;
}
TextInRect* TextInRect::setWidth(float value) {
    shape.setSize(sf::Vector2f(
        value, getHeight()
    ));
    return this;
}
TextInRect* TextInRect::setHeight(float value) {
    shape.setSize(sf::Vector2f(
        getWidth(), value
    ));
    return this;
}
float TextInRect::getLeft() const { return shape.getPosition().x; }
float TextInRect::getTop() const { return shape.getPosition().y; }
float TextInRect::getWidth() const { return shape.getSize().x; }
float TextInRect::getHeight() const { return shape.getSize().y; }

TextInRect* TextInRect::fit() {
    sf::FloatRect text_bounding = text.getGlobalBounds();
    text.setPosition(
        getLeft() + padding,
        getTop() + padding
    );
    shape.setSize(sf::Vector2f(
        text_bounding.width + padding*2 + 2.f,
        text_bounding.height + padding*3 + 3.f
    ));
    return this;
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

sf::FloatRect TextInRect::getGlobalBounds() const {
    return shape.getGlobalBounds();
}

const sf::Color TextInRect::hover_color = sf::Color(255, 255, 255, 60);
const sf::Color TextInRect::active_color = sf::Color(255, 255, 255, 60);
