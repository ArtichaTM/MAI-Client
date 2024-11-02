#include <string>
#include <sstream>

#include <SFML/Graphics.hpp>

#include "config.hpp"
#include "./button.hpp"

Button::Button(
    float x,
    float y,
    const std::string& _text,
    const std::string& font_path,
    sf::Color main_color
) : TextInRect(x, y, _text, font_path, main_color)
{}

void Button::setOnClick(std::function<void()> _on_click) {
    on_click = _on_click;
}

Button::operator std::string() const {
    std::stringstream ss;
    ss << "<Button with text \"" << (std::string) text.getString() << "\">";
    return ss.str();
}

void Button::keyPressed(const sf::Mouse::Button& button) {
    TextInRect::keyPressed(button);
    on_click();
}
