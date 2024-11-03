#pragma once
#include <SFML/Graphics.hpp>

struct SFBase :
    public sf::Drawable,
    public sf::Transformable
{
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const = 0;
    virtual SFBase* setLeft(float) = 0;
    virtual SFBase* setTop(float) = 0;
    virtual SFBase* setWidth(float) = 0;
    virtual SFBase* setHeight(float) = 0;
    virtual float getLeft() const = 0;
    virtual float getTop() const = 0;
    virtual float getWidth() const = 0;
    virtual float getHeight() const = 0;

    virtual sf::FloatRect getGlobalBounds() const;

    virtual void handleEvent(const sf::Event&);
    bool isVectorInBounds(const sf::Vector2f&) const;

protected:
    virtual void mouseMoved(const sf::Vector2f&);
    virtual void keyPressed(const sf::Mouse::Button&);
    virtual void keyReleased(const sf::Mouse::Button&);
    virtual void keyPressed(const sf::Event::KeyEvent&);
    virtual void keyReleased(const sf::Event::KeyEvent&);
};
