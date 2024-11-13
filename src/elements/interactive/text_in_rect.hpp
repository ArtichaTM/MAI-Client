#pragma once
#include <string>
#include <SFML/Graphics.hpp>

#include "elements/base.hpp"

struct TextInRect : public SFBase {
    TextInRect(
        float x,
        float y,
        const std::string& text,
        const std::string& font_path,
        sf::Color main_color
    );
    ~TextInRect();
    operator std::string() const;

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    sf::FloatRect getGlobalBounds() const override;
    void move(const float left, const float top) override;

    void fit() override;
    TextInRect* setPadding(float);
    void recalculateColor();

    void setActive(bool);
    bool getActive();
    void setHovering(bool);
    bool getHovering();

private:
    bool active = false;
    bool hovering = false;
    static const sf::Color hover_color;
    static const sf::Color active_color;

protected:
    sf::RectangleShape shape;
    sf::Font font;
    sf::Text text;
    sf::Color* current_color = nullptr;
    sf::Color start_color;
    float padding = 6.f;

    void keyPressed(const sf::Mouse::Button&) override;
    void mouseMoved(const sf::Vector2f&) override;
};
