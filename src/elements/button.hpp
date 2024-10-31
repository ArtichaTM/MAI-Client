#pragma once
#include <string>
#include <SFML/Graphics.hpp>

#include "base.hpp"

struct Button : public SFBase {
    sf::RectangleShape shape;
    sf::Font font;
    sf::Text text;

    Button(
        float x,
        float y,
        const std::string& text,
        const std::string& font_path,
        sf::Color color
    );

    void handleEvent(const sf::Event&);
    void draw(sf::RenderWindow& window) const;
    bool isMouseOver();

private:
    static constexpr float padding = 4.f;
    static const sf::Color hover_color;
};
