#pragma once
#include <SFML/Graphics.hpp>

class SFBase
    :
    public sf::Drawable,
    public sf::Transformable
{
public:
    virtual void handleEvent(const sf::Event&) = 0;
    virtual void draw(sf::RenderWindow& window) = 0;
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
};
