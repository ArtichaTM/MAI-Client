#pragma once
#include <string>
#include <SFML/Graphics.hpp>

#include "./text_in_rect.hpp"

class Button : public TextInRect {
    std::function<void()> on_click;
public:
    Button(
        float x,
        float y,
        const std::string& text,
        const std::string& font_path,
        sf::Color main_color
    );
    operator std::string() const;
    void setOnClick(std::function<void()>);

    void keyPressed(const sf::Mouse::Button&) override;
};
