#include "./base.hpp"
#include "base.hpp"

BaseList::BaseList(float _left, float _top)
 : left(_left), top(_top),
 current_left(_left), current_top(_top)
{}

BaseList::~BaseList()
{
    for (SFBase* el : elements) delete el;
}

void BaseList::draw(sf::RenderTarget &target, sf::RenderStates states) const
{
    for (SFBase* el : elements) el->draw(target, states);
}

void BaseList::handleEvent(const sf::Event& e)
{
    for (SFBase* el : elements) el->handleEvent(e);
}

void BaseList::fit() {}

void BaseList::move(const float left, const float top)
{
    for (SFBase* el : elements) el->move(left, top);
}

const std::vector<SFBase *> BaseList::getElements() const { return elements; }

BaseList* BaseList::addElement(SFBase* element)
{
    element->fit();
    element->move(current_left, current_top);
    updateCurrentPosition(element);
    elements.push_back(element);
    return this;
}
