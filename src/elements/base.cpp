#include <SFML/Graphics.hpp>

#include "base.hpp"

void SFBase::draw(sf::RenderTarget &target, sf::RenderStates states) const {
    throw std::logic_error("U can't call this method");
}
