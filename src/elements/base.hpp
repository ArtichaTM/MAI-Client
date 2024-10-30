#pragma once
#include <SFML/Graphics.hpp>

class SFBase
    :
    public sf::Drawable,
    public sf::Transformable
{
public:
    virtual void handleEvent(const sf::Event&) = 0;
    virtual void draw(sf::RenderWindow& window) const = 0;
};
