#pragma once
#include <string>
#include <SFML/Graphics.hpp>

#include "elements/base.hpp"

struct TextInRect : public SFBase {
    sf::RectangleShape shape;
    sf::Font font;
    sf::Text text;
    sf::Color* current_color = nullptr;
    sf::Color start_color;
    float padding = 6.f;

    TextInRect(
        float x,
        float y,
        const std::string& text,
        const std::string& font_path,
        sf::Color main_color
    );
    ~TextInRect();
    operator std::string() const;

    sf::FloatRect getGlobalBounds() const override;
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    TextInRect* setLeft(float) override;
    TextInRect* setTop(float) override;
    TextInRect* setWidth(float) override;
    TextInRect* setHeight(float) override;
    float getLeft() const override;
    float getTop() const override;
    float getWidth() const override;
    float getHeight() const override;

    TextInRect* fit();
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
    void keyPressed(const sf::Mouse::Button&) override;
    void mouseMoved(const sf::Vector2f&) override;
};
