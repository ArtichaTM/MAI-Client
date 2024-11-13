#include "./vertical.hpp"
#include "vertical.hpp"

void VerticalList::updateCurrentPosition(SFBase* element) {
    current_top += element->getGlobalBounds().height;
}

VerticalList::VerticalList(float left, float top) : BaseList(left, top) {}

sf::FloatRect VerticalList::getGlobalBounds() const
{
    sf::FloatRect rect(left, top, 0, 0);
    for (SFBase* el : elements) {
        const sf::FloatRect el_bounds(el->getGlobalBounds());
        rect.height += el_bounds.height;
        if (rect.width < el_bounds.width) rect.width = el_bounds.width;
    }
    return rect;
}