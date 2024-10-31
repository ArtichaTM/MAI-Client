#pragma once
#include <string>
#include <SFML/Graphics.hpp>

#include "../base.hpp"

struct Button : public SFBase {
    sf::RectangleShape shape;
    sf::Font font;
    sf::Text text;
    sf::Color* current_color = nullptr;
    sf::Color start_color;

    Button(
        float x,
        float y,
        const std::string& text,
        const std::string& font_path,
        sf::Color main_color
    );
    ~Button();

    void handleEvent(const sf::Event&);
    void draw(sf::RenderWindow& window) override;
    bool isMouseOver();
    void recalculateColor();

    void setActive(bool);
    void setHovering(bool);

private:
    bool active = false;
    bool hovering = false;
    static constexpr float padding = 6.f;
    static const sf::Color hover_color;
    static const sf::Color active_color;

};
