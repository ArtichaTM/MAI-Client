#include <SFML/Graphics.hpp>

#include "config.hpp"
#include "./base.hpp"
#include "base.hpp"

bool SFBase::isVectorInBounds(const sf::Vector2f& vec) const {
    return getGlobalBounds().contains(vec);
}

void SFBase::move(const sf::FloatRect& rect)
{
    move(rect.left, rect.top);
}

void SFBase::handleEvent(const sf::Event &event)
{
    switch (event.type) {
        case sf::Event::MouseButtonPressed: {
            if (!isVectorInBounds(
                ROOT_WINDOW->mapPixelToCoords(
                    sf::Mouse::getPosition(*ROOT_WINDOW)
                )
            )) return;
            keyPressed(event.mouseButton.button);
            break;
        }
        case sf::Event::MouseButtonReleased: {
            if (!isVectorInBounds(
                ROOT_WINDOW->mapPixelToCoords(
                    sf::Mouse::getPosition(*ROOT_WINDOW)
                )
            )) return;
            keyReleased(event.mouseButton.button);
            break;
        }
        case sf::Event::MouseMoved: {
            mouseMoved(
                ROOT_WINDOW->mapPixelToCoords(
                    sf::Mouse::getPosition(*ROOT_WINDOW)
                )
            );
            break;
        }
        case sf::Event::KeyPressed: {
            keyPressed(event.key);
            break;
        }
        case sf::Event::KeyReleased: {
            keyReleased(event.key);
            break;
        }
    }
}

void SFBase::mouseMoved(const sf::Vector2f &) {}
void SFBase::keyPressed(const sf::Mouse::Button&) {}
void SFBase::keyReleased(const sf::Mouse::Button&) {}
void SFBase::keyPressed(const sf::Event::KeyEvent&) {}
void SFBase::keyReleased(const sf::Event::KeyEvent&) {}
