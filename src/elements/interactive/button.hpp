#pragma once
#include <string>
#include <SFML/Graphics.hpp>

#include "./text_in_rect.hpp"

class Button : public TextInRect {
    std::function<void(Button*)> on_click;
    bool activatingDisabled = false;
public:
    Button(
        float x,
        float y,
        const std::string& text,
        const std::string& font_path,
        sf::Color main_color
    );
    operator std::string() const;
    Button* setOnClick(std::function<void(Button*)>);
    Button* toggleActivatable();

    void keyPressed(const sf::Mouse::Button&) override;
};
