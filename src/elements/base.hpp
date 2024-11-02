#pragma once
#include <SFML/Graphics.hpp>

struct SFBase :
    public sf::Drawable,
    public sf::Transformable
{
    virtual void handleEvent(const sf::Event&);
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const = 0;
    virtual sf::FloatRect getGlobalBounds() const = 0;
    bool isVectorInBounds(const sf::Vector2f&) const;
protected:
    virtual void mouseMoved(const sf::Vector2f&);
    virtual void keyPressed(const sf::Mouse::Button&);
    virtual void keyReleased(const sf::Mouse::Button&);
    virtual void keyPressed(const sf::Event::KeyEvent&);
    virtual void keyReleased(const sf::Event::KeyEvent&);
};
