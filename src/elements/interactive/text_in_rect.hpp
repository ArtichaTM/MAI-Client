#pragma once
#include <string>
#include <SFML/Graphics.hpp>

#include "../base.hpp"

struct TextInRect : public SFBase {
    sf::RectangleShape shape;
    sf::Font font;
    sf::Text text;
    sf::Color* current_color = nullptr;
    sf::Color start_color;

    TextInRect(
        float x,
        float y,
        const std::string& text,
        const std::string& font_path,
        sf::Color main_color
    );
    ~TextInRect();
    operator std::string() const;

    bool isVectorInBounds(const sf::Vector2f&) override;
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    void recalculateColor();

    void setActive(bool);
    bool getActive();
    void setHovering(bool);
    bool getHovering();

private:
    bool active = false;
    bool hovering = false;
    static constexpr float padding = 6.f;
    static const sf::Color hover_color;
    static const sf::Color active_color;

protected:
    void keyPressed(const sf::Mouse::Button&) override;
    void mouseMoved(const sf::Vector2f&) override;
};
