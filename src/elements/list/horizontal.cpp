#include "./horizontal.hpp"
#include "horizontal.hpp"

void HorizontalList::updateCurrentPosition(SFBase* element) {
    current_left += element->getGlobalBounds().width;
}

HorizontalList::HorizontalList(float left, float top) : BaseList(left, top) {}

sf::FloatRect HorizontalList::getGlobalBounds() const
{
    sf::FloatRect rect(left, top, 0, 0);
    for (SFBase* el : elements) {
        const sf::FloatRect el_bounds(el->getGlobalBounds());
        rect.width += el_bounds.width;
        if (rect.height < el_bounds.height) rect.height = el_bounds.height;
    }
    return rect;
}