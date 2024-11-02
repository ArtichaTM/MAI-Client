#pragma once
#include <SFML/Graphics.hpp>

struct SFBase :
    public sf::Drawable,
    public sf::Transformable
{
    virtual void handleEvent(const sf::Event&);
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    virtual bool isVectorInBounds(const sf::Vector2f&) = 0;
protected:
    virtual void mouseMoved(const sf::Vector2f&);
    virtual void keyPressed(const sf::Mouse::Button&);
    virtual void keyReleased(const sf::Mouse::Button&);
    virtual void keyPressed(const sf::Event::KeyEvent&);
    virtual void keyReleased(const sf::Event::KeyEvent&);
};
