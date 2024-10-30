#pragma once
#include <string>
#include <SFML/Graphics.hpp>


class Button {
public:
    Button(
        float x,
        float y,
        float width,
        float height,
        const std::string& text
    );

    void draw(sf::RenderWindow&);
    bool isMouseOver(sf::Vector2i);
    void onClick();
    void update(sf::Vector2i);

private:
    sf::RectangleShape buttonShape;
    sf::Font buttonFont;
    sf::Text buttonText;
};
